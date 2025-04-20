import asyncio
import time
import uuid
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from pymodbus.client import AsyncModbusTcpClient
from pymodbus.exceptions import ModbusException

class PlasmaDataCollector:
    def __init__(self, ip="192.168.1.10", sample_rate=100, buffer_size=5000):
        self.ip = ip
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._client = None
        self._buffer = []
        self._buffer_lock = asyncio.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._running = False

    async def connect(self):
        self._client = AsyncModbusTcpClient(self.ip)
        await self._client.connect()
        self._running = True

    async def disconnect(self):
        self._running = False
        if self._client:
            await self._client.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.flush()
        await self.disconnect()

    async def _read_batch(self):
        try:
            rr = await self._client.read_holding_registers(0, 2, unit=1)
            if rr.isError():
                raise ModbusException(rr)
            return rr.registers[0] * 0.1, rr.registers[1]
        except ModbusException as e:
            self._logger.error(f"Read error: {e}")
            await self._reconnect()
            return None, None

    async def _reconnect(self):
        await self.disconnect()
        await asyncio.sleep(1)
        await self.connect()

    async def flush(self):
        async with self._buffer_lock:
            if self._buffer:
                await self._write_buffer()

    async def _write_buffer(self):
        table = pa.Table.from_pydict({
            "timestamp": [x[0] for x in self._buffer],
            "plasma_temp": [x[1] for x in self._buffer],
            "gamma_flux": [x[2] for x in self._buffer]
        })
        
        filename = f"data/plasma_{uuid.uuid4().hex}.parquet"
        try:
            pq.write_table(table, filename)
            self._logger.info(f"Wrote {len(self._buffer)} records to {filename}")
        except Exception as e:
            self._logger.error(f"Write failed: {e}")

    async def collect(self):
        expected_interval = 1.0 / self.sample_rate
        while self._running:
            start_time = time.perf_counter()
            
            temp, gamma = await self._read_batch()
            if temp is None:
                continue
                
            timestamp = pa.timestamp('us').cast(time.time_ns() // 1000)
            
            async with self._buffer_lock:
                self._buffer.append((timestamp, temp, gamma))
                if len(self._buffer) >= self.buffer_size:
                    await self._write_buffer()
                    self._buffer.clear()

            # Adaptif uyku süresi
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, expected_interval - elapsed)
            await asyncio.sleep(sleep_time)
