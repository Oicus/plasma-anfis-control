from pymodbus.client import AsyncModbusTcpClient
import pandas as pd
import asyncio

class PlasmaDataCollector:
    def __init__(self, ip="192.168.1.10"):
        self.client = AsyncModbusTcpClient(ip)
        
    async def collect(self, duration=3600, sample_rate=100):
        """Collects data for 1 hour at 100Hz"""
        data = []
        for _ in range(int(duration*sample_rate)):
            try:
                temp = (await self.client.read_holding_registers(0)).registers[0]*0.1
                gamma = (await self.client.read_input_registers(1)).registers[0]
                data.append([pd.Timestamp.now(), temp, gamma])
                await asyncio.sleep(1/sample_rate)
            except Exception as e:
                print(f"Error: {e}")
        
        df = pd.DataFrame(data, columns=["timestamp", "plasma_temp", "gamma_flux"])
        df.to_parquet("data/raw_sensor_data.parquet")
        return df

# Usage
async def main():
    collector = PlasmaDataCollector()
    await collector.collect(duration=60)  # 1-minute test

asyncio.run(main())
