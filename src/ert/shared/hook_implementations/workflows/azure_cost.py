from __future__ import annotations
import os
from collections import defaultdict
from datetime import timedelta
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from ert.config import ErtScript
import requests

class _Compute(BaseModel):
    location: str
    sku: Annotated[str, Field(alias="vmSize")]


class _Instance(BaseModel):
    compute: _Compute


class _Item(BaseModel):
    region: Annotated[str, Field(alias="armRegionName")]
    unit: Annotated[str, Field(alias="unitOfMeasure")]
    price: Annotated[float, Field(alias="retailPrice")]
    name: Annotated[str, Field(alias="armSkuName")]
    product_name: Annotated[str, Field(alias="productName")]
    sku_name: Annotated[str, Field(alias="skuName")]


class _PricingResponse(BaseModel):
    items: Annotated[List[_Item], Field(alias="Items")]
    next_page: Annotated[Optional[str], Field(alias="NextPageLink")]


class AzureCost(ErtScript):
    def run(self, *args: str) -> None:
        with open("cost-report", "a") as f:
            runpaths = self.parse_runpaths(args[0])

            ens = self.ensemble or self.storage.get_ensemble_by_name("default")

            f.write(f"Computing costs for {ens.name}:")
            ens_total = 0.0
            for iens in range(ens.ensemble_size):
                try:
                    real = ens.get_realization(iens)
                except KeyError:
                    continue

                runpath = runpaths[ens.iteration][iens]
                with open(os.path.join(runpath, ".azure-instance")) as g:
                    instance = _Instance.model_validate_json(g.read())

                delta = real.end_time - real.start_time
                ens_total += self.get_pricing(instance.compute.location, instance.compute.sku, delta)
            f.write(f" {ens_total} USD\n")


    def parse_runpaths(self, path: str) -> dict[int, dict[int, str]]:
        data: dict[int, dict[int, str]] = defaultdict(dict)
        with open(path) as f:
            for line in f:
                iens, rpath, name, iter = line.split()
                data[int(iter)][int(iens)] = rpath
        return data


    @lru_cache()
    def get_hourly(self, location: str, sku: str) -> float:
        url = f"https://prices.azure.com/api/retail/prices?currencyCode=USD&$filter=armRegionName eq '{location}' and armSkuName eq '{sku}' and type eq 'Consumption'"
        with requests.Session() as session:
            response = session.get(url)
            pricing = _PricingResponse.model_validate_json(response.content)
            for item in pricing.items:
                if "Windows" in item.product_name:
                    continue
                if "Spot" in item.sku_name or "Low Priority" in item.sku_name:
                    continue
                return item.price
        raise NotImplementedError("I don't know what to do in this case")


    def get_pricing(self, location: str, sku: str, delta: timedelta) -> float:
        return (delta.days * 24 + delta.seconds / 3600) * self.get_hourly(location, sku)
