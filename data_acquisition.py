from pymodbus.client import AsyncModbusTcpClient
from pymodbus.exceptions import ModbusIOException, ConnectionException
import pandas as pd
import asyncio
import time
from contextlib import asynccontextmanager

class PlasmaDataCollector:
    def __init__(self, ip="192.168.1.10", sample_rate=100):
        self.ip = ip
        self._client = None
        self._sample_interval = 1 / sample_rate  # örnekleme aralığı (saniye)
        self._buffer = []
        self._buffer_lock = asyncio.Lock()

    @property
    async def client(self):
        # Otomatik bağlantı kontrolü
        if not self._client or not self._client.connected:
            self._client = AsyncModbusTcpClient(self.ip)
            await self._client.connect()
        return self._client

    async def _reset_connection(self):
        if self._client:
            await self._client.close()
        self._client = AsyncModbusTcpClient(self.ip)
        await self._client.connect()

    async def _read_sensors(self):
        client = await self.client
        try:
            # Aynı anda sıcaklık ve gamma ölçümleri alınır
            temp, gamma = await asyncio.gather(
                client.read_holding_registers(0, 1, unit=1),
                client.read_input_registers(1, 1, unit=1)
            )
            return (
                temp.registers[0] * 0.1,  # 10 tabanlı hassasiyet çarpanı
                gamma.registers[0]
            )
        except (ModbusIOException, ConnectionException) as e:
            print(f"[ERROR] Modbus read failed: {e}")
            await self._reset_connection()
            return None, None
        except Exception as e:
            print(f"[ERROR] Unknown exception: {e}")
            return None, None

    async def _data_writer(self):
        while True:
            async with self._buffer_lock:
                if len(self._buffer) >= 5000:
                    df = pd.DataFrame(self._buffer, columns=["timestamp", "plasma_temp", "gamma_flux"])
                    df.to_parquet(f"data/plasma_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.parquet")
                    self._buffer.clear()
            await asyncio.sleep(1)

    async def collect(self, duration=3600):
        start_time = time.monotonic()
        writer_task = asyncio.create_task(self._data_writer())

        try:
            while (time.monotonic() - start_time) < duration:
                loop_start = time.perf_counter()

                temp, gamma = await self._read_sensors()
                if temp is None:
                    continue  # veri geçersizse tekrar dene

                timestamp = pd.Timestamp.now().tz_localize('UTC')

                async with self._buffer_lock:
                    self._buffer.append([timestamp, temp, gamma])

                elapsed = time.perf_counter() - loop_start
                await asyncio.sleep(max(0, self._sample_interval - elapsed))

        except Exception as e:
            print(f"[CRITICAL] Collection loop interrupted: {e}")
        finally:
            writer_task.cancel()

        # Kalan verileri yaz
        async with self._buffer_lock:
            if self._buffer:
                df = pd.DataFrame(self._buffer, columns=["timestamp", "plasma_temp", "gamma_flux"])
                df.to_parquet(f"data/plasma_final_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.parquet")
                return df

