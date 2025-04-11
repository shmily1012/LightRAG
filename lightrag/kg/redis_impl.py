import os
from typing import Any, final
from dataclasses import dataclass
import pipmaster as pm
import configparser
from contextlib import asynccontextmanager
import traceback
if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis, ConnectionPool  # type: ignore
from redis.exceptions import RedisError, ConnectionError  # type: ignore
from redis.backoff import ExponentialBackoff  # type: ignore
from redis.retry import Retry  # type: ignore
from lightrag.utils import logger

from lightrag.base import BaseKVStorage
import json
import asyncio
from itertools import islice


config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Constants for Redis connection pool
MAX_CONNECTIONS = 50
SOCKET_TIMEOUT = 5.0
SOCKET_CONNECT_TIMEOUT = 3.0
POOL_HEALTH_CHECK_INTERVAL = 30  # seconds
MAX_RETRIES = 3
RETRY_ON_TIMEOUT = True
RETRY_ON_ERROR = True

# Concurrency control settings
MAX_CONCURRENT_OPERATIONS = 10  # Maximum number of concurrent Redis operations
BATCH_SIZE = 100  # Number of items to process in a single batch


@final
@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
        redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        
        # Configure retry strategy
        retry = Retry(
            ExponentialBackoff(),
            MAX_RETRIES,
            retry_on_timeout=RETRY_ON_TIMEOUT,
            retry_on_error=RETRY_ON_ERROR
        )
        
        # Create a connection pool with enhanced configuration
        self._pool = ConnectionPool.from_url(
            redis_url,
            max_connections=MAX_CONNECTIONS,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT,
            socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
            health_check_interval=POOL_HEALTH_CHECK_INTERVAL,
            retry=retry
        )
        
        self._redis = Redis(connection_pool=self._pool)
        self._monitor_task = None
        
        # Initialize semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_OPERATIONS)
        
        logger.info(
            f"Initialized Redis connection pool for {self.namespace} with:"
            f"\n- Max connections: {MAX_CONNECTIONS}"
            f"\n- Health check interval: {POOL_HEALTH_CHECK_INTERVAL}s"
            f"\n- Max retries: {MAX_RETRIES}"
            f"\n- Max concurrent operations: {MAX_CONCURRENT_OPERATIONS}"
            f"\n- Batch size: {BATCH_SIZE}"
        )
        
        # Start connection pool monitoring
        self._start_pool_monitor()

    @staticmethod
    def _batch_items(items: list, batch_size: int):
        """Helper method to split items into batches"""
        iterator = iter(items)
        return iter(lambda: list(islice(iterator, batch_size)), [])

    async def _process_batch(self, batch: list[str]) -> list[Any]:
        """Process a batch of IDs with concurrency control"""
        async with self._semaphore:
            async with self._get_redis_connection() as redis:
                try:
                    pipe = redis.pipeline()
                    for id in batch:
                        pipe.get(f"{self.namespace}:{id}")
                    results = await pipe.execute()
                    return [json.loads(result) if result else None for result in results]
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in batch processing: {e}")
                    return [None] * len(batch)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple items by their IDs with batching and concurrency control"""
        if not ids:
            return []

        all_results = []
        batches = list(self._batch_items(ids, BATCH_SIZE))
        
        # Process batches concurrently but with controlled parallelism
        tasks = [self._process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results from all batches
        for batch_result in batch_results:
            all_results.extend(batch_result)
            
        return all_results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter keys with batching and concurrency control"""
        if not keys:
            return set()

        async def process_key_batch(batch: list[str]) -> list[bool]:
            async with self._semaphore:
                async with self._get_redis_connection() as redis:
                    pipe = redis.pipeline()
                    for key in batch:
                        pipe.exists(f"{self.namespace}:{key}")
                    return await pipe.execute()

        all_results = []
        key_list = list(keys)
        batches = list(self._batch_items(key_list, BATCH_SIZE))
        
        # Process batches concurrently but with controlled parallelism
        tasks = [process_key_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten and process results
        for batch, exists_batch in zip(batches, batch_results):
            for key, exists in zip(batch, exists_batch):
                if not exists:
                    all_results.append(key)

        return set(all_results)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data with batching and concurrency control"""
        if not data:
            return

        logger.info(f"Inserting {len(data)} items to {self.namespace}")

        async def process_upsert_batch(batch_data: dict[str, dict[str, Any]]) -> None:
            async with self._semaphore:
                async with self._get_redis_connection() as redis:
                    try:
                        pipe = redis.pipeline()
                        for k, v in batch_data.items():
                            pipe.set(f"{self.namespace}:{k}", json.dumps(v))
                        await pipe.execute()
                    except ValueError as e:
                        traceback.print_exc()
                        logger.error(f"JSON encode error during batch upsert: {e}")
                        raise

        # Split data into batches
        batches = []
        current_batch = {}
        for i, (k, v) in enumerate(data.items()):
            current_batch[k] = v
            if (i + 1) % BATCH_SIZE == 0:
                batches.append(current_batch)
                current_batch = {}
        if current_batch:
            batches.append(current_batch)

        # Process batches concurrently but with controlled parallelism
        tasks = [process_upsert_batch(batch) for batch in batches]
        await asyncio.gather(*tasks)

        # Update IDs in the original data
        for k in data:
            data[k]["_id"] = k

    async def _monitor_pool(self):
        """Monitor connection pool statistics periodically"""
        while True:
            try:
                # Get pool statistics
                used_connections = len(self._pool._used_connections)
                available_connections = len(self._pool._available_connections)
                
                logger.debug(
                    f"Redis pool stats for {self.namespace}:"
                    f"\n- Used connections: {used_connections}"
                    f"\n- Available connections: {available_connections}"
                    f"\n- Total: {used_connections + available_connections}/{MAX_CONNECTIONS}"
                )
                
                # Check for potential connection leaks
                if used_connections > MAX_CONNECTIONS * 0.8:  # 80% threshold
                    logger.warning(
                        f"High number of used connections in {self.namespace}: "
                        f"{used_connections}/{MAX_CONNECTIONS}"
                    )
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in pool monitoring: {e}")
                await asyncio.sleep(60)  # Continue monitoring even after error

    def _start_pool_monitor(self):
        """Start the connection pool monitoring task"""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitor_pool())

    @asynccontextmanager
    async def _get_redis_connection(self):
        """Safe context manager for Redis operations."""
        try:
            yield self._redis
        except ConnectionError as e:
            traceback.print_exc()
            logger.error(f"Redis connection error in {self.namespace}: {e}")
            raise
        except RedisError as e:
            traceback.print_exc()
            logger.error(f"Redis operation error in {self.namespace}: {e}")
            raise
        except Exception as e:
            traceback.print_exc()
            logger.error(
                f"Unexpected error in Redis operation for {self.namespace}: {e}"
            )
            raise

    async def close(self):
        """Close the Redis connection pool to prevent resource leaks."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        if hasattr(self, "_redis") and self._redis:
            await self._redis.close()
            await self._pool.disconnect()
            logger.debug(f"Closed Redis connection pool for {self.namespace}")

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure Redis resources are cleaned up when exiting context."""
        await self.close()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get a single item by ID with proper connection management and concurrency control.
        
        Args:
            id: The ID of the item to retrieve
            
        Returns:
            The item data as a dictionary, or None if not found
        """
        async with self._semaphore:  # Control concurrent access
            async with self._get_redis_connection() as redis:
                try:
                    # Use pipeline for efficiency even with single operation
                    pipe = redis.pipeline()
                    pipe.get(f"{self.namespace}:{id}")
                    [data] = await pipe.execute()
                    
                    if not data:
                        return None
                        
                    try:
                        return json.loads(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error for id {id}: {e}")
                        return None
                except RedisError as e:
                    logger.error(f"Redis error in get_by_id for {id}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error in get_by_id for {id}: {e}")
                    traceback.print_exc()
                    return None

    async def index_done_callback(self) -> None:
        # Redis handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete entries with specified IDs"""
        if not ids:
            return

        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for id in ids:
                pipe.delete(f"{self.namespace}:{id}")

            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(
                f"Deleted {deleted_count} of {len(ids)} entries from {self.namespace}"
            )

    async def drop_cache_by_modes(self, modes: list[str] | None = None) -> bool:
        """Delete specific records from storage by by cache mode

        Importance notes for Redis storage:
        1. This will immediately delete the specified cache modes from Redis

        Args:
            modes (list[str]): List of cache mode to be drop from storage

        Returns:
             True: if the cache drop successfully
             False: if the cache drop failed
        """
        if not modes:
            return False

        try:
            await self.delete(modes)
            return True
        except Exception:
            return False

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all keys under the current namespace with batching and concurrency control.
        
        The operation is performed in batches to prevent overwhelming the Redis server
        and includes proper concurrency control using semaphores.
        
        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        async with self._get_redis_connection() as redis:
            try:
                # First, scan for all keys matching the pattern
                pattern = f"{self.namespace}:*"
                all_keys = []
                cursor = 0
                
                # Use SCAN instead of KEYS for large datasets
                while True:
                    cursor, keys = await redis.scan(
                        cursor=cursor, 
                        match=pattern, 
                        count=BATCH_SIZE
                    )
                    all_keys.extend(keys)
                    if cursor == 0:
                        break

                if not all_keys:
                    logger.info(f"No keys found to drop in {self.namespace}")
                    return {"status": "success", "message": "no keys to drop"}

                total_keys = len(all_keys)
                logger.info(f"Found {total_keys} keys to delete in {self.namespace}")

                async def process_delete_batch(batch_keys: list[str]) -> int:
                    """Process a batch of keys for deletion with concurrency control"""
                    async with self._semaphore:
                        async with self._get_redis_connection() as batch_redis:
                            pipe = batch_redis.pipeline()
                            for key in batch_keys:
                                pipe.delete(key)
                            results = await pipe.execute()
                            return sum(1 for r in results if r)

                # Split keys into batches
                batches = list(self._batch_items(all_keys, BATCH_SIZE))
                deleted_counts = []

                # Process batches with controlled concurrency
                tasks = [process_delete_batch(batch) for batch in batches]
                batch_results = await asyncio.gather(*tasks)
                total_deleted = sum(batch_results)

                # Log progress after each batch is complete
                logger.info(
                    f"Successfully dropped {total_deleted}/{total_keys} keys "
                    f"from {self.namespace}"
                )

                return {
                    "status": "success",
                    "message": f"{total_deleted} keys dropped"
                }

            except Exception as e:
                error_msg = f"Error dropping keys from {self.namespace}: {str(e)}"
                logger.error(error_msg)
                traceback.print_exc()
                return {"status": "error", "message": error_msg}
