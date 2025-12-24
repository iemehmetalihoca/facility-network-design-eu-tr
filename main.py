from __future__ import annotations


from folium.plugins import MarkerCluster, Fullscreen, MiniMap, MeasureControl, LocateControl

import argparse
import copy
import io
import json
import math
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import yaml
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, LpBinary,
    LpStatus, value, PULP_CBC_CMD
)

import folium


ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data_raw"
DATA_OUT = ROOT / "data_out"
OUTPUTS = ROOT / "outputs"
CFG_PATH = ROOT / "config.yaml"


def ensure_dirs():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def report_uncovered_cafes(cafes: pd.DataFrame, hubs: pd.DataFrame, sla_km: float, out_csv: Path | None = None):
    rows = []
    for i, c in cafes.iterrows():
        best_km = float("inf")
        best_j = None
        for j, h in hubs.iterrows():
            km = haversine_km(float(c["lat"]), float(c["lon"]), float(h["lat"]), float(h["lon"]))
            if km < best_km:
                best_km, best_j = km, j

        if best_km > sla_km:
            rows.append({
                "cafe_idx": int(i),
                "cafe_city": str(c.get("city", "")),
                "cafe_country": str(c.get("country", "")),
                "nearest_hub_city": str(hubs.loc[best_j, "city"]) if best_j is not None else "",
                "nearest_hub_country": str(hubs.loc[best_j, "country"]) if best_j is not None else "",
                "nearest_km": float(best_km),
            })

    if not rows:
        print(f"[OK] All cafes have at least 1 candidate hub within SLA={sla_km:.1f} km.")
        return 0.0

    print("\n[UN-COVERED] Cafes with NO candidate hub within SLA:")
    for r in rows:
        print(f"- cafe_idx={r['cafe_idx']}: {r['cafe_city']} ({r['cafe_country']}) "
              f"nearest={r['nearest_hub_city']} ({r['nearest_hub_country']}) "
              f"dist={r['nearest_km']:.1f} km")

    need_sla = max(r["nearest_km"] for r in rows)
    print(f"[SUGGEST] Increase max_service_km to at least: {need_sla:.1f} km")

    if out_csv is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
        print("[Wrote]", out_csv)

    return float(need_sla)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def minmax_norm(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if abs(mx - mn) < 1e-12:
        return s * 0.0
    return (s - mn) / (mx - mn)


import re


def load_country_cost_proxy(data_raw: Path) -> Dict[str, float]:
    cinfo = load_country_info(data_raw / "countryInfo.txt")
    iso3_to_iso2 = dict(zip(cinfo["ISO3"], cinfo["ISO"]))
    name_to_iso2 = dict(zip(cinfo["Country"].str.lower(), cinfo["ISO"]))

    lab = pd.read_csv(data_raw / "labour_cost_country.csv")
    lab = lab[
        (lab["freq"] == "A") &
        (lab["unit"] == "EUR") &
        (lab["lcstruct"] == "D11") &
        (lab["nace_r2"] == "B-S_X_O")
    ].copy()
    lab["TIME_PERIOD"] = pd.to_numeric(lab["TIME_PERIOD"], errors="coerce")
    lab["OBS_VALUE"] = pd.to_numeric(lab["OBS_VALUE"], errors="coerce")
    lab_latest = (
        lab.sort_values("TIME_PERIOD")
           .groupby("geo")
           .tail(1)[["geo", "OBS_VALUE"]]
           .rename(columns={"geo": "ISO2", "OBS_VALUE": "labour_eur"})
    )

    cit = pd.read_csv(data_raw / "cit_rate_country.csv")
    cit["TIME_PERIOD"] = pd.to_numeric(cit["TIME_PERIOD"], errors="coerce")
    cit["OBS_VALUE"] = pd.to_numeric(cit["OBS_VALUE"], errors="coerce")
    cit = cit[cit["MEASURE"].isin(["CIT", "CIT_C"])].copy()
    cit["ISO2"] = cit["REF_AREA"].map(iso3_to_iso2)
    cit = cit.dropna(subset=["ISO2"])

    cit_latest = cit.sort_values("TIME_PERIOD").groupby(["ISO2", "MEASURE"]).tail(1)
    cit_piv = cit_latest.pivot(index="ISO2", columns="MEASURE", values="OBS_VALUE").reset_index()
    cit_piv["cit_rate"] = cit_piv["CIT"] if "CIT" in cit_piv.columns else None
    if "cit_rate" not in cit_piv or cit_piv["cit_rate"].isna().all():
        cit_piv["cit_rate"] = cit_piv["CIT_C"] if "CIT_C" in cit_piv.columns else None
    cit_piv = cit_piv[["ISO2", "cit_rate"]]

    rent = pd.read_csv(data_raw / "rent_country.csv")
    country_col = "country" if "country" in rent.columns else ("Country" if "Country" in rent.columns else rent.columns[0])

    xcols = [c for c in rent.columns if re.fullmatch(r"x\d+", str(c))]
    rent[xcols] = rent[xcols].apply(pd.to_numeric, errors="coerce")
    z = (rent[xcols] - rent[xcols].mean()) / rent[xcols].std(ddof=0)
    rent["rent_score"] = z.mean(axis=1)

    rent["ISO2"] = rent[country_col].astype(str).str.lower().map(name_to_iso2)
    rent = rent.dropna(subset=["ISO2"])[["ISO2", "rent_score"]]

    feat = lab_latest.merge(cit_piv, on="ISO2", how="left").merge(rent, on="ISO2", how="left")

    for col in ["labour_eur", "cit_rate", "rent_score"]:
        feat[col] = feat[col].fillna(feat[col].mean())
        feat[col + "_n"] = minmax_norm(feat[col])

    w_r, w_l, w_t = 0.45, 0.45, 0.10
    feat["fixed_cost_proxy_ext"] = 1.0 + w_r * feat["rent_score_n"] + w_l * feat["labour_eur_n"] + w_t * feat["cit_rate_n"]

    return feat.set_index("ISO2")["fixed_cost_proxy_ext"].to_dict()


import unicodedata


def _norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_labour_tax_country(data_raw: Path) -> tuple[dict[str, float], dict[str, float]]:
    lab = pd.read_csv(data_raw / "labour_cost_country.csv")
    lab = lab[
        (lab["freq"] == "A") &
        (lab["unit"] == "EUR") &
        (lab["lcstruct"] == "D11") &
        (lab["nace_r2"] == "B-S_X_O")
    ].copy()
    lab["TIME_PERIOD"] = pd.to_numeric(lab["TIME_PERIOD"], errors="coerce")
    lab["OBS_VALUE"] = pd.to_numeric(lab["OBS_VALUE"], errors="coerce")
    lab_latest = (
        lab.sort_values("TIME_PERIOD")
           .groupby("geo")
           .tail(1)[["geo", "OBS_VALUE"]]
           .rename(columns={"geo": "ISO2", "OBS_VALUE": "labour_eur"})
    )

    cinfo = load_country_info(data_raw / "countryInfo.txt")
    iso3_to_iso2 = dict(zip(cinfo["ISO3"], cinfo["ISO"]))

    cit = pd.read_csv(data_raw / "cit_rate_country.csv")
    cit["TIME_PERIOD"] = pd.to_numeric(cit["TIME_PERIOD"], errors="coerce")
    cit["OBS_VALUE"] = pd.to_numeric(cit["OBS_VALUE"], errors="coerce")
    cit = cit[cit["MEASURE"].isin(["CIT", "CIT_C"])].copy()
    cit["ISO2"] = cit["REF_AREA"].map(iso3_to_iso2)
    cit = cit.dropna(subset=["ISO2"])

    cit_latest = cit.sort_values("TIME_PERIOD").groupby(["ISO2", "MEASURE"]).tail(1)
    cit_piv = cit_latest.pivot(index="ISO2", columns="MEASURE", values="OBS_VALUE").reset_index()
    if "CIT" in cit_piv.columns:
        cit_piv["cit_rate"] = cit_piv["CIT"]
    elif "CIT_C" in cit_piv.columns:
        cit_piv["cit_rate"] = cit_piv["CIT_C"]
    else:
        cit_piv["cit_rate"] = None
    cit_piv = cit_piv[["ISO2", "cit_rate"]]

    feat = lab_latest.merge(cit_piv, on="ISO2", how="left")
    for col in ["labour_eur", "cit_rate"]:
        feat[col] = feat[col].fillna(feat[col].mean())
        feat[col + "_n"] = minmax_norm(feat[col])

    labour_n = feat.set_index("ISO2")["labour_eur_n"].to_dict()
    tax_n = feat.set_index("ISO2")["cit_rate_n"].to_dict()
    return labour_n, tax_n


def load_rent_scores_city_country(data_raw: Path) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    cinfo = load_country_info(data_raw / "countryInfo.txt")
    name_to_iso2 = dict(zip(cinfo["Country"].str.lower(), cinfo["ISO"]))

    rent_city_path = data_raw / "rent_city.csv"
    if not rent_city_path.exists():
        return {}, {}

    rc = pd.read_csv(rent_city_path)
    if "city" not in rc.columns or "country" not in rc.columns:
        return {}, {}

    xcols = [c for c in rc.columns if re.fullmatch(r"x\d+", str(c))]
    rc[xcols] = rc[xcols].apply(pd.to_numeric, errors="coerce")

    z = (rc[xcols] - rc[xcols].mean()) / rc[xcols].std(ddof=0)
    rc["rent_score_city"] = z.mean(axis=1)

    rc["ISO2"] = rc["country"].astype(str).str.lower().map(name_to_iso2)
    rc = rc.dropna(subset=["ISO2"]).copy()
    rc["city_key"] = rc["city"].map(_norm_key)

    city_score = {(row.city_key, row.ISO2): float(row.rent_score_city) for row in rc.itertuples(index=False)}
    country_score = rc.groupby("ISO2")["rent_score_city"].mean().to_dict()
    return city_score, country_score


def build_fixed_cost_per_candidate(
    candidates: pd.DataFrame,
    data_raw: Path,
    *,
    base_hub_cost: float,
    base_dc_cost: float,
    w_r: float = 0.45,
    w_l: float = 0.45,
    w_t: float = 0.10,
) -> tuple[dict[int, float], dict[int, float], pd.DataFrame]:
    df = candidates.copy().reset_index(drop=True)
    df["city_key"] = df["city"].map(_norm_key)

    labour_n, tax_n = load_labour_tax_country(data_raw)
    city_score, rent_country = load_rent_scores_city_country(data_raw)

    warn_city_fallback = set()
    warn_global_fallback = set()

    df["rent_score_city"] = float("nan")
    df["rent_score_country"] = float("nan")
    df["rent_source_used"] = ""

    for i in range(len(df)):
        city = str(df.loc[i, "city"])
        iso2 = str(df.loc[i, "country"])
        ck = df.loc[i, "city_key"]

        city_val = city_score.get((ck, iso2))
        country_val = rent_country.get(iso2)

        if city_val is not None:
            df.at[i, "rent_score_city"] = float(city_val)
            df.at[i, "rent_source_used"] = "city"
        else:
            if country_val is not None:
                warn_city_fallback.add((city, iso2))
                df.at[i, "rent_source_used"] = "country"
            else:
                warn_global_fallback.add((city, iso2))
                df.at[i, "rent_source_used"] = "global"

        if country_val is not None:
            df.at[i, "rent_score_country"] = float(country_val)

    if warn_city_fallback:
        print("\n[WARN] rent_city.csv içinde city-level kira bulunamadı; ülke ortalaması kullanıldı:")
        for city, iso2 in sorted(warn_city_fallback):
            print(f"  - {city} ({iso2}) -> country mean rent_score")

    if warn_global_fallback:
        print("\n[WARN] rent_city.csv / rent_country.csv içinde kira bilgisi hiç yok; global ortalama kullanıldı:")
        for city, iso2 in sorted(warn_global_fallback):
            print(f"  - {city} ({iso2}) -> global mean rent_score")

    df["rent_score_used"] = df["rent_score_city"]
    miss = df["rent_score_used"].isna()
    df.loc[miss, "rent_score_used"] = df.loc[miss, "rent_score_country"]

    if rent_country:
        global_rent_mean = float(pd.Series(list(rent_country.values())).mean())
    else:
        tmp = df["rent_score_used"]
        if tmp.notna().any():
            global_rent_mean = float(tmp.dropna().mean())
        else:
            global_rent_mean = 0.0

    df["rent_score_used"] = df["rent_score_used"].fillna(global_rent_mean)
    df["rent_n"] = minmax_norm(df["rent_score_used"])

    labour_mean = float(pd.Series(list(labour_n.values())).mean()) if labour_n else 0.5
    tax_mean = float(pd.Series(list(tax_n.values())).mean()) if tax_n else 0.5

    missing_lab_countries = set()
    missing_tax_countries = set()

    lab_vals = []
    tax_vals = []

    for i in range(len(df)):
        iso2 = str(df.loc[i, "country"])

        if iso2 in labour_n:
            lab_vals.append(float(labour_n[iso2]))
        else:
            lab_vals.append(labour_mean)
            missing_lab_countries.add(iso2)

        if iso2 in tax_n:
            tax_vals.append(float(tax_n[iso2]))
        else:
            tax_vals.append(tax_mean)
            missing_tax_countries.add(iso2)

    df["labour_n"] = lab_vals
    df["tax_n"] = tax_vals

    if missing_lab_countries:
        print("\n[WARN] labour_cost_country.csv içinde olmayan ülkeler; labour için global ortalama kullanıldı:")
        print("  - " + ", ".join(sorted(missing_lab_countries)))

    if missing_tax_countries:
        print("\n[WARN] cit_rate_country.csv içinde olmayan ülkeler; vergi için global ortalama kullanıldı:")
        print("  - " + ", ".join(sorted(missing_tax_countries)))

    df["fixed_mult"] = 1.0 + w_r * df["rent_n"] + w_l * df["labour_n"] + w_t * df["tax_n"]
    df["fixed_cost_hub"] = base_hub_cost * df["fixed_mult"]
    df["fixed_cost_dc"] = base_dc_cost * df["fixed_mult"]

    hub_fixed = {j: float(df.loc[j, "fixed_cost_hub"]) for j in range(len(df))}
    dc_fixed = {j: float(df.loc[j, "fixed_cost_dc"]) for j in range(len(df))}

    return hub_fixed, dc_fixed, df


GEONAMES_BASE = "https://download.geonames.org/export/dump"
CITIES_ZIP = "cities15000.zip"
COUNTRY_INFO = "countryInfo.txt"


def download_if_missing(url: str, dest: Path, timeout=90):
    if dest.exists():
        return
    print(f"Downloading: {url}")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)


def load_country_info(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            rows.append(line.rstrip("\n").split("\t"))
    cols = [
        "ISO", "ISO3", "ISONum", "FIPS", "Country", "Capital", "Area", "Population", "Continent",
        "tld", "CurrencyCode", "CurrencyName", "Phone", "PostalFmt", "PostalRegex", "Languages",
        "geonameid", "neighbours", "EqFips",
    ]
    return pd.DataFrame(rows, columns=cols)


def parse_cities15000(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            txt = [n for n in z.namelist() if n.endswith(".txt")][0]
            raw = z.read(txt)
        bio = io.BytesIO(raw)
        df = pd.read_csv(
            bio, sep="\t", header=None, low_memory=False,
            names=["geonameid","name","asciiname","alternatenames","lat","lon","fclass","fcode",
                   "country_code","cc2","admin1","admin2","admin3","admin4","population",
                   "elevation","dem","timezone","moddate"],
            usecols=["geonameid","name","country_code","lat","lon","population"],
        )
    else:
        df = pd.read_csv(
            path, sep="\t", header=None, low_memory=False,
            names=["geonameid","name","asciiname","alternatenames","lat","lon","fclass","fcode",
                   "country_code","cc2","admin1","admin2","admin3","admin4","population",
                   "elevation","dem","timezone","moddate"],
            usecols=["geonameid","name","country_code","lat","lon","population"],
        )

    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    return df


def step_top100():
    ensure_dirs()

    cities_txt = DATA_RAW / "cities15000.txt"
    cities_zip = DATA_RAW / "cities15000.zip"
    country_info = DATA_RAW / "countryInfo.txt"

    if not country_info.exists():
        raise FileNotFoundError("data_raw/countryInfo.txt bulunamadı...")

    if cities_txt.exists():
        cities_src = cities_txt
    elif cities_zip.exists():
        cities_src = cities_zip
    else:
        raise FileNotFoundError("data_raw içinde cities15000.txt veya .zip yok...")

    cinfo = load_country_info(country_info)
    europe_iso = set(cinfo.loc[cinfo["Continent"] == "EU", "ISO"].tolist())
    europe_iso.add("TR")

    exclude = {"RU"}
    europe_iso = {iso for iso in europe_iso if iso not in exclude}

    cities = parse_cities15000(cities_src)
    cities = cities[cities["country_code"].isin(europe_iso)].copy()
    cities = cities.sort_values("population", ascending=False).head(100)

    out = cities.rename(columns={"name": "city", "country_code": "country"})[
        ["geonameid", "city", "country", "lat", "lon", "population"]
    ].reset_index(drop=True)

    out_path = DATA_OUT / "top100_cities.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Wrote:", out_path)
    print(out.head(5))


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]


def build_count_query(lat: float, lon: float, radius_m: int, clauses: List[Tuple[str, str]]) -> str:
    parts = []
    for k, v in clauses:
        parts.append(f'node["{k}"="{v}"](around:{radius_m},{lat},{lon});')
        parts.append(f'way["{k}"="{v}"](around:{radius_m},{lat},{lon});')
        parts.append(f'relation["{k}"="{v}"](around:{radius_m},{lat},{lon});')
    return "[out:json][timeout:25];(\n" + "\n".join(parts) + "\n);out count;"


def parse_count(resp_json: dict) -> int:
    try:
        el = resp_json["elements"][0]
        tags = el.get("tags", {})
        if "total" in tags:
            return int(tags["total"])
        s = 0
        for v in tags.values():
            try:
                s += int(v)
            except Exception:
                pass
        return int(s)
    except Exception:
        return 0


def overpass_post(query: str) -> dict:
    last_err = None
    for ep in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(ep, data=query.encode("utf-8"), timeout=75)
            if r.status_code in (429, 502, 504):
                raise RuntimeError(f"Overpass busy: {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_err}")


def request_with_backoff(query: str, max_tries: int = 7) -> dict:
    for t in range(max_tries):
        try:
            return overpass_post(query)
        except Exception as e:
            wait = min(90, (2 ** t) + random.random())
            print(f"[retry {t+1}/{max_tries}] {e} -> sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Overpass failed after retries.")


def step_poi():
    ensure_dirs()
    cfg = load_config()
    radius = int(cfg.get("poi_radius_m", 10000))

    cities_path = DATA_OUT / "top100_cities.csv"
    if not cities_path.exists():
        raise FileNotFoundError("Run: python project.py top100  (top100_cities.csv not found)")

    df = pd.read_csv(cities_path)

    cache_dir = DATA_OUT / "poi_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    CAT = {
        "poi_student": [
            ("amenity", "university"),
            ("amenity", "college"),
            ("amenity", "school"),
        ],
        "poi_activity": [
            ("shop", "mall"),
            ("amenity", "cinema"),
            ("shop", "supermarket"),
            ("leisure", "park"),
        ],
        "poi_logistics": [
            ("highway", "motorway_junction"),
            ("aeroway", "aerodrome"),
            ("man_made", "pier"),
            ("harbour", "yes"),
        ],
        "poi_transit": [
            ("railway", "station"),
            ("public_transport", "station"),
            ("highway", "bus_stop"),
            ("amenity", "bus_station"),
        ],
    }

    rows = []
    for _, r in df.iterrows():
        gid = int(r["geonameid"])
        lat = float(r["lat"])
        lon = float(r["lon"])

        cache_file = cache_dir / f"{gid}_r{radius}.json"
        if cache_file.exists():
            rec = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            rec = {"geonameid": gid}
            for key, clauses in CAT.items():
                q = build_count_query(lat, lon, radius, clauses)
                js = request_with_backoff(q)
                rec[key] = parse_count(js)
                time.sleep(0.2)
            cache_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

        rows.append(rec)

    poi = pd.DataFrame(rows)

    merged = df.merge(poi, on="geonameid", how="left")

    merged["pop_norm"] = minmax_norm(merged["population"])
    merged["student_norm"] = minmax_norm(merged["poi_student"].fillna(0))
    merged["activity_norm"] = minmax_norm(merged["poi_activity"].fillna(0))
    merged["logistics_norm"] = minmax_norm(merged["poi_logistics"].fillna(0))
    merged["transit_norm"] = minmax_norm(merged["poi_transit"].fillna(0))

    merged["demand_factor"] = 1 + 0.6 * merged["student_norm"] + 0.2 * merged["activity_norm"]
    merged["hub_score"] = 0.7 * merged["logistics_norm"] + 0.3 * merged["transit_norm"]
    merged["fixed_cost_proxy"] = 1 + 0.8 * merged["pop_norm"] + 0.4 * merged["activity_norm"]

    out_path = DATA_OUT / "poi_scores_100.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print("Wrote:", out_path)
    print(merged[["city", "country", "poi_student", "poi_activity", "poi_logistics", "poi_transit", "hub_score"]].head(5))


def _pick_with_country_cap(df_sorted: pd.DataFrame, n: int, max_per_country: int) -> pd.DataFrame:
    chosen = []
    ccount = {}
    for _, r in df_sorted.iterrows():
        iso2 = str(r["country"])
        if ccount.get(iso2, 0) >= max_per_country:
            continue
        chosen.append(r)
        ccount[iso2] = ccount.get(iso2, 0) + 1
        if len(chosen) >= n:
            break
    return pd.DataFrame(chosen) if chosen else df_sorted.head(0).copy()


def step_candidates():
    def _spatial_thin(df_sorted: pd.DataFrame, radius_km: float) -> pd.DataFrame:
        if df_sorted.empty:
            return df_sorted.copy()

        keep_rows = []
        kept_coords: list[tuple[float, float]] = []
        for _, r in df_sorted.iterrows():
            lat = float(r["lat"]); lon = float(r["lon"])
            ok = True
            for klat, klon in kept_coords:
                if haversine_km(lat, lon, klat, klon) <= radius_km:
                    ok = False
                    break
            if ok:
                keep_rows.append(r)
                kept_coords.append((lat, lon))
        return pd.DataFrame(keep_rows).reset_index(drop=True)

    ensure_dirs()
    cfg = load_config()
    src = DATA_OUT / "poi_scores_100.csv"
    if not src.exists():
        raise FileNotFoundError("Run: python project.py poi  (poi_scores_100.csv not found)")

    df = pd.read_csv(src)

    cand_n = int(cfg.get("candidate_hubs_n", 60))
    cand_n = max(10, min(cand_n, len(df)))

    cs = (cfg.get("candidate_selection", {}) or {})
    w = (cs.get("score_weights", {}) or {})
    w_infra = float(w.get("infra", 0.55))
    w_demand = float(w.get("demand", 0.15))
    w_cost = float(w.get("cost", 0.30))

    region_min = (cs.get("region_min", {}) or {})
    min_tr = int(region_min.get("TR", 0))
    min_eu = int(region_min.get("EU", 0))
    max_per_country = int(cs.get("max_per_country", 8))

    spatial_thin_km = float(cs.get("spatial_thin_km", 30.0))

    base_hub_cost = 1.0
    base_dc_cost = 1.0
    _, _, df_costed = build_fixed_cost_per_candidate(
        df[["geonameid", "city", "country", "lat", "lon", "hub_score", "population", "demand_factor"]].copy(),
        DATA_RAW,
        base_hub_cost=base_hub_cost,
        base_dc_cost=base_dc_cost,
        w_r=float((cfg.get("fixed_cost_weights", {}) or {}).get("rent", 0.45)),
        w_l=float((cfg.get("fixed_cost_weights", {}) or {}).get("labour", 0.45)),
        w_t=float((cfg.get("fixed_cost_weights", {}) or {}).get("tax", 0.10)),
    )

    df_costed["pop_norm"] = minmax_norm(pd.to_numeric(df_costed["population"], errors="coerce").fillna(0.0))
    demand_proxy = (
        (0.3 + 0.7 * df_costed["pop_norm"]) *
        pd.to_numeric(df_costed["demand_factor"], errors="coerce").fillna(1.0)
    )
    df_costed["demand_n"] = minmax_norm(demand_proxy)

    df_costed["hub_score_n"] = minmax_norm(pd.to_numeric(df_costed["hub_score"], errors="coerce").fillna(0.0))

    df_costed["fixed_mult_n"] = minmax_norm(pd.to_numeric(df_costed["fixed_mult"], errors="coerce").fillna(1.0))

    df_costed["candidate_score"] = (
        w_infra * df_costed["hub_score_n"]
        + w_demand * df_costed["demand_n"]
        - w_cost * df_costed["fixed_mult_n"]
    )

    df_costed["region"] = df_costed["country"].astype(str).apply(lambda x: "TR" if x == "TR" else "EU")

    df_sorted = df_costed.sort_values("candidate_score", ascending=False).reset_index(drop=True)

    before_n = len(df_sorted)
    df_sorted = _spatial_thin(df_sorted, radius_km=spatial_thin_km)
    print(f"[INFO] Spatial thinning applied (r={spatial_thin_km:.1f} km): {before_n} -> {len(df_sorted)}")

    if len(df_sorted) < cand_n:
        print(f"[WARN] After thinning, pool < cand_n ({len(df_sorted)} < {cand_n}). "
              f"Consider reducing candidate_selection.spatial_thin_km or increasing candidate_hubs_n.")

    picked_parts = []

    if min_tr > 0:
        tr_sorted = df_sorted[df_sorted["region"] == "TR"].copy()
        picked_parts.append(_pick_with_country_cap(tr_sorted, min_tr, max_per_country))

    if min_eu > 0:
        eu_sorted = df_sorted[df_sorted["region"] == "EU"].copy()
        picked_parts.append(_pick_with_country_cap(eu_sorted, min_eu, max_per_country))

    picked = pd.concat(picked_parts, ignore_index=True) if picked_parts else df_sorted.head(0).copy()

    picked_ids = set(picked["geonameid"].astype(int).tolist())
    remaining = df_sorted[~df_sorted["geonameid"].astype(int).isin(picked_ids)].copy()

    need = cand_n - len(picked)
    if need > 0:
        ccount = picked["country"].astype(str).value_counts().to_dict()
        chosen_more = []
        for _, r in remaining.iterrows():
            iso2 = str(r["country"])
            if ccount.get(iso2, 0) >= max_per_country:
                continue
            chosen_more.append(r)
            ccount[iso2] = ccount.get(iso2, 0) + 1
            if len(chosen_more) >= need:
                break
        if chosen_more:
            picked = pd.concat([picked, pd.DataFrame(chosen_more)], ignore_index=True)

    cand = picked.reset_index(drop=True).copy()

    out = DATA_OUT / "candidates_50.csv"
    cand.to_csv(out, index=False, encoding="utf-8")
    print("Wrote:", out)

    diag = DATA_OUT / "candidate_pool_diagnostics.csv"
    df_sorted.to_csv(diag, index=False, encoding="utf-8")
    print("Wrote:", diag)

    print(cand[["city", "country", "region", "hub_score", "candidate_score", "fixed_mult", "rent_source_used"]].head(10))


def step_optimize():
    ensure_dirs()
    cfg = load_config()

    cafes_path = DATA_OUT / "poi_scores_100.csv"
    hubs_path  = DATA_OUT / "candidates_50.csv"

    if not cafes_path.exists() or not hubs_path.exists():
        raise FileNotFoundError("Need poi_scores_100.csv and candidates_50.csv. Run: top100 -> poi -> candidates")

    cafes = pd.read_csv(cafes_path).reset_index(drop=True)
    hubs_raw = pd.read_csv(hubs_path).reset_index(drop=True)

    max_service_km = float(cfg["max_service_km"])
    min_hubs = int(cfg["min_hubs"])
    max_hubs = int(cfg["max_hubs"])

    base_hub_cost = float(cfg.get("fixed_cost_scale", 1000))

    demand_multiplier = float(cfg.get("demand_multiplier", 1.0))
    enable_co2 = bool(cfg.get("enable_co2", False))
    co2_cost = float(cfg.get("co2_cost_per_ton_km", 0.0))

    one_dc_per_product = bool(cfg.get("one_dc_per_product", False))

    w = cfg.get("fixed_cost_weights", {}) or {}
    w_r = float(w.get("rent", 0.45))
    w_l = float(w.get("labour", 0.45))
    w_t = float(w.get("tax", 0.10))

    dc_cfg = cfg.get("dc", {}) or {}
    min_dcs = int(dc_cfg.get("min_dcs", 2))
    max_dcs = int(dc_cfg.get("max_dcs", 4))
    dc_cost_multiplier = float(dc_cfg.get("dc_cost_multiplier", 6.0))
    base_dc_cost = float(dc_cfg.get("base_dc_cost", base_hub_cost * dc_cost_multiplier))

    min_dcs_tr = int(dc_cfg.get("min_dcs_tr", 1))
    min_dcs_eu = int(dc_cfg.get("min_dcs_eu", 1))
    enforce_region = bool(dc_cfg.get("enforce_region", True))

    cap_cfg = cfg.get("capacity", {}) or {}
    hub_cap_ton = float(cap_cfg.get("hub_max_total_ton", 1e12))
    dc_cap_ton = float(cap_cfg.get("dc_max_total_ton", 1e12))
    hub_max_cafes = int(cap_cfg.get("hub_max_cafes", 10**9))

    products = list(cfg["products"].keys())
    out_cost = {p: float(cfg["products"][p]["out_cost_per_ton_km"]) for p in products}
    in_cost = {p: float(cfg["products"][p]["in_cost_per_ton_km"]) for p in products}
    base_ton = {p: float(cfg["products"][p]["base_ton_per_city"]) for p in products}

    cafes["pop_norm"] = minmax_norm(cafes["population"])
    cafes["base_weight"] = (0.3 + 0.7 * cafes["pop_norm"]) * cafes["demand_factor"] * demand_multiplier
    demand = {(i, p): base_ton[p] * float(cafes.loc[i, "base_weight"])
              for i in range(len(cafes)) for p in products}

    hubs = hubs_raw[["geonameid", "city", "country", "lat", "lon", "hub_score"]].copy()
    hub_fixed, dc_fixed, hubs_costed = build_fixed_cost_per_candidate(
        hubs,
        DATA_RAW,
        base_hub_cost=base_hub_cost,
        base_dc_cost=base_dc_cost,
        w_r=w_r, w_l=w_l, w_t=w_t,
    )

    infra_cfg = cfg.get("infra_discount", {}) or {}
    disc_hub = float(infra_cfg.get("hub", 0.0))
    disc_dc = float(infra_cfg.get("dc", 0.0))

    hubs_costed["hub_score_n"] = minmax_norm(pd.to_numeric(hubs_costed["hub_score"], errors="coerce").fillna(0.0))

    hubs_costed["fixed_cost_hub_eff"] = hubs_costed["fixed_cost_hub"] * (1.0 - disc_hub * hubs_costed["hub_score_n"])
    hubs_costed["fixed_cost_dc_eff"] = hubs_costed["fixed_cost_dc"] * (1.0 - disc_dc * hubs_costed["hub_score_n"])

    hubs_costed["fixed_cost_hub_eff"] = hubs_costed["fixed_cost_hub_eff"].clip(lower=0.70 * hubs_costed["fixed_cost_hub"])
    hubs_costed["fixed_cost_dc_eff"] = hubs_costed["fixed_cost_dc_eff"].clip(lower=0.70 * hubs_costed["fixed_cost_dc"])

    hub_fixed = {j: float(hubs_costed.loc[j, "fixed_cost_hub_eff"]) for j in range(len(hubs_costed))}
    dc_fixed = {j: float(hubs_costed.loc[j, "fixed_cost_dc_eff"]) for j in range(len(hubs_costed))}

    rent_counts = hubs_costed["rent_source_used"].value_counts(dropna=False).to_dict()
    print("[INFO] Rent source counts:", rent_counts)

    hubs_costed.to_csv(OUTPUTS / "candidate_costs.csv", index=False, encoding="utf-8")

    feasible_ij: List[Tuple[int, int]] = []
    dist_ij: Dict[Tuple[int, int], float] = {}
    edges_by_i: Dict[int, List[int]] = {i: [] for i in range(len(cafes))}
    edges_by_j: Dict[int, List[int]] = {j: [] for j in range(len(hubs))}

    for i in range(len(cafes)):
        lat_i, lon_i = float(cafes.loc[i, "lat"]), float(cafes.loc[i, "lon"])
        for j in range(len(hubs)):
            lat_j, lon_j = float(hubs.loc[j, "lat"]), float(hubs.loc[j, "lon"])
            km = haversine_km(lat_i, lon_i, lat_j, lon_j)
            if km <= max_service_km:
                feasible_ij.append((i, j))
                dist_ij[(i, j)] = km
                edges_by_i[i].append(j)
                edges_by_j[j].append(i)

    bad = [i for i in range(len(cafes)) if len(edges_by_i[i]) == 0]
    if bad:
        need_sla = report_uncovered_cafes(
            cafes=cafes,
            hubs=hubs,
            sla_km=max_service_km,
            out_csv=OUTPUTS / "uncovered_cafes.csv",
        )
        raise RuntimeError(
            f"Infeasible: {len(bad)} cafes have no hub within max_service_km={max_service_km}. "
            f"Try increasing max_service_km to >= {need_sla:.1f} km or expand candidate hubs."
        )

    cover = LpProblem("MinHubsCover", LpMinimize)
    y2 = LpVariable.dicts("y2", range(len(hubs)), 0, 1, LpBinary)
    cover += lpSum(y2[j] for j in range(len(hubs)))
    for i in range(len(cafes)):
        cover += lpSum(y2[j] for j in edges_by_i[i]) >= 1, f"Cover_{i}"
    cover.solve(PULP_CBC_CMD(msg=False, timeLimit=60))
    cover_status = LpStatus[cover.status]
    print("Cover status:", cover_status)
    if cover_status not in ("Optimal", "Feasible"):
        raise RuntimeError("Cover model did not solve; try increasing timeLimit.")

    min_needed = int(math.ceil(value(cover.objective) - 1e-9))
    print("Minimum hubs needed for coverage:", min_needed)
    if min_needed > max_hubs:
        raise RuntimeError(
            f"Infeasible: need at least {min_needed} hubs to cover all cafes, but max_hubs={max_hubs}. "
            f"Increase max_hubs or relax max_service_km."
        )

    dc_allowed = set(dc_cfg.get("allowed_countries", []))
    if dc_allowed:
        K = [
            j for j in range(len(hubs))
            if str(hubs.loc[j, "country"]) in dc_allowed
        ]
        if not K:
            raise RuntimeError(
                f"No DC candidates remain after applying allowed_countries={dc_allowed}. "
                f"Check dc.allowed_countries or candidate pool."
            )
    else:
        K = list(range(len(hubs)))

    dist_kj: Dict[Tuple[int, int], float] = {}
    for k in K:
        lat_k, lon_k = float(hubs.loc[k, "lat"]), float(hubs.loc[k, "lon"])
        for j in range(len(hubs)):
            lat_j, lon_j = float(hubs.loc[j, "lat"]), float(hubs.loc[j, "lon"])
            dist_kj[(k, j)] = haversine_km(lat_k, lon_k, lat_j, lon_j)

    M_p = {p: 1.05 * sum(demand[(i, p)] for i in range(len(cafes))) for p in products}

    prob = LpProblem("DC_Hub_Cafe_Network", LpMinimize)

    y = LpVariable.dicts("openHub", range(len(hubs)), 0, 1, LpBinary)
    u = LpVariable.dicts("openDC", K, 0, 1, LpBinary)
    x = LpVariable.dicts("assign", feasible_ij, 0, 1, LpBinary)
    f = LpVariable.dicts("flow", (K, range(len(hubs)), products), lowBound=0)

    if one_dc_per_product:
        z = LpVariable.dicts("oneDC", (K, range(len(hubs)), products), 0, 1, LpBinary)

    obj_fixed_hub = lpSum(hub_fixed[j] * y[j] for j in range(len(hubs)))
    obj_fixed_dc = lpSum(dc_fixed[k] * u[k] for k in K)

    obj_out = lpSum(
        out_cost[p] * dist_ij[(i, j)] * demand[(i, p)] * x[(i, j)]
        for (i, j) in feasible_ij for p in products
    )

    obj_in = lpSum(
        in_cost[p] * dist_kj[(k, j)] * f[k][j][p]
        for k in K for j in range(len(hubs)) for p in products
    )

    obj_co2 = 0
    if enable_co2 and co2_cost > 0:
        obj_co2 = (
            lpSum(co2_cost * dist_kj[(k, j)] * f[k][j][p]
                  for k in K for j in range(len(hubs)) for p in products)
            +
            lpSum(co2_cost * dist_ij[(i, j)] * demand[(i, p)] * x[(i, j)]
                  for (i, j) in feasible_ij for p in products)
        )

    prob += obj_fixed_hub + obj_fixed_dc + obj_in + obj_out + obj_co2

    for i in range(len(cafes)):
        prob += lpSum(x[(i, j)] for j in edges_by_i[i]) == 1, f"AssignOnce_{i}"

    for (i, j) in feasible_ij:
        prob += x[(i, j)] <= y[j], f"OpenLink_{i}_{j}"

    prob += lpSum(y[j] for j in range(len(hubs))) >= min_hubs, "MinHubs"
    prob += lpSum(y[j] for j in range(len(hubs))) <= max_hubs, "MaxHubs"

    prob += lpSum(u[k] for k in K) >= min_dcs, "MinDCs"
    prob += lpSum(u[k] for k in K) <= max_dcs, "MaxDCs"

    if enforce_region:
        tr_k = [k for k in K if str(hubs.loc[k, "country"]) == "TR"]
        eu_k = [k for k in K if str(hubs.loc[k, "country"]) != "TR"]
        if tr_k:
            prob += lpSum(u[k] for k in tr_k) >= min_dcs_tr, "MinDC_TR"
        if eu_k:
            prob += lpSum(u[k] for k in eu_k) >= min_dcs_eu, "MinDC_EU"

    for j in range(len(hubs)):
        for p in products:
            rhs = lpSum(demand[(i, p)] * x[(i, j)] for i in edges_by_j[j])
            prob += lpSum(f[k][j][p] for k in K) == rhs, f"FlowBal_{j}_{p}"

    for j in range(len(hubs)):
        total_ton_j = lpSum(
            demand[(i, p)] * x[(i, j)]
            for i in edges_by_j[j]
            for p in products
        )
        prob += total_ton_j <= hub_cap_ton * y[j], f"HubCapTon_{j}"

    if hub_max_cafes < 10**9:
        for j in range(len(hubs)):
            prob += lpSum(
                x[(i, j)] for i in edges_by_j[j]
            ) <= hub_max_cafes * y[j], f"HubCapCafe_{j}"

    for k in K:
        total_ton_k = lpSum(
            f[k][j][p] for j in range(len(hubs)) for p in products
        )
        prob += total_ton_k <= dc_cap_ton * u[k], f"DcCapTon_{k}"

    for k in K:
        for j in range(len(hubs)):
            for p in products:
                prob += f[k][j][p] <= M_p[p] * u[k], f"LinkFU_{k}_{j}_{p}"

    if one_dc_per_product:
        for j in range(len(hubs)):
            for p in products:
                prob += lpSum(z[k][j][p] for k in K) == y[j], f"OneDC_{j}_{p}"

                for k in K:
                    prob += z[k][j][p] <= u[k], f"LinkZU_{k}_{j}_{p}"
                    prob += f[k][j][p] <= M_p[p] * z[k][j][p], f"LinkFZ_{k}_{j}_{p}"

    solver = PULP_CBC_CMD(msg=True, timeLimit=1800)
    prob.solve(solver)

    print("Pulp raw status code:", prob.status)
    print("Pulp status:", LpStatus[prob.status])

    status = LpStatus[prob.status]
    if status not in ("Optimal", "Feasible"):
        raise RuntimeError("No feasible/optimal solution. Try relaxing SLA or changing hub/DC bounds.")

    open_hubs = [j for j in range(len(hubs)) if value(y[j]) > 0.5]
    open_dcs = [k for k in K if value(u[k]) > 0.5]

    assigns = []
    for (i, j) in feasible_ij:
        if value(x[(i, j)]) > 0.5:
            assigns.append((i, j, dist_ij[(i, j)]))

    hubs_open = hubs_costed.loc[open_hubs].copy()
    hubs_open["hub_idx"] = open_hubs
    hubs_open.to_csv(OUTPUTS / "opened_hubs.csv", index=False, encoding="utf-8")

    dcs_open = hubs_costed.loc[open_dcs].copy()
    dcs_open["hub_idx"] = open_dcs
    dcs_open.to_csv(OUTPUTS / "opened_dcs.csv", index=False, encoding="utf-8")

    flow_rows = []
    for k in K:
        if value(u[k]) <= 0.5:
            continue
        for j in range(len(hubs)):
            if value(y[j]) <= 0.5:
                continue
            for p in products:
                ton = value(f[k][j][p])
                if ton is None:
                    continue
                ton = float(ton)
                if ton <= 1e-6:
                    continue

                km = float(dist_kj[(k, j)])

                flow_rows.append({
                    "dc_idx": int(k),
                    "hub_idx": int(j),
                    "product": str(p),
                    "ton": ton,
                    "dist_km": km,
                })

    flow_df = pd.DataFrame(flow_rows)
    flow_path = OUTPUTS / "dc_hub_flows.csv"
    flow_df.to_csv(flow_path, index=False, encoding="utf-8")
    print("Wrote:", flow_path)

    assigns_df = pd.DataFrame(assigns, columns=["cafe_idx", "hub_idx", "dist_km"])
    cafes_idx = cafes.reset_index().rename(columns={"index": "cafe_idx"})
    hubs_idx = hubs_costed.reset_index().rename(columns={"index": "hub_idx"})
    assigns_df = assigns_df.merge(cafes_idx, on="cafe_idx", how="left", suffixes=("", "_cafe"))
    assigns_df = assigns_df.merge(hubs_idx, on="hub_idx", how="left", suffixes=("_cafe", "_hub"))
    assigns_df.to_csv(OUTPUTS / "assignments.csv", index=False, encoding="utf-8")

    obj_fixed_hub_v = float(value(obj_fixed_hub))
    obj_fixed_dc_v = float(value(obj_fixed_dc))
    obj_in_v = float(value(obj_in))
    obj_out_v = float(value(obj_out))
    obj_co2_v = float(value(obj_co2)) if obj_co2 != 0 else 0.0

    summary = {
        "status": status,
        "objective": float(value(prob.objective)),
        "obj_fixed_hub": obj_fixed_hub_v,
        "obj_fixed_dc": obj_fixed_dc_v,
        "obj_inbound": obj_in_v,
        "obj_outbound": obj_out_v,
        "obj_co2": obj_co2_v,
        "hubs_opened": int(len(open_hubs)),
        "dcs_opened": int(len(open_dcs)),
        "avg_cafe_to_hub_km": float(assigns_df["dist_km"].mean()),
        "max_cafe_to_hub_km": float(assigns_df["dist_km"].max()),
        "max_service_km": max_service_km,
        "fixed_cost_scale": base_hub_cost,
        "min_hubs": min_hubs,
        "max_hubs": max_hubs,
        "min_dcs": min_dcs,
        "max_dcs": max_dcs,
        "base_hub_cost": base_hub_cost,
        "base_dc_cost": base_dc_cost,
        "one_dc_per_product": one_dc_per_product,
        "enable_co2": enable_co2,
    }

    summary["rent_src_city"] = int(rent_counts.get("city", 0))
    summary["rent_src_country"] = int(rent_counts.get("country", 0))
    summary["rent_src_global"] = int(rent_counts.get("global", 0))
    summary["candidate_n"] = int(len(hubs_costed))

    pd.DataFrame([summary]).to_csv(OUTPUTS / "solution_summary.csv", index=False, encoding="utf-8")
    print("Wrote:", OUTPUTS / "solution_summary.csv")


def step_map():
    ensure_dirs()
    cfg = load_config()

    cafes_path = DATA_OUT / "poi_scores_100.csv"
    hubs_open_path  = OUTPUTS / "opened_hubs.csv"
    dcs_open_path   = OUTPUTS / "opened_dcs.csv"
    assigns_path    = OUTPUTS / "assignments.csv"
    candidates_path = DATA_OUT / "candidates_50.csv"
    flows_path = OUTPUTS / "dc_hub_flows.csv"

    if not cafes_path.exists():
        raise FileNotFoundError("Missing poi_scores_100.csv. Run: python project.py poi")
    if not hubs_open_path.exists():
        raise FileNotFoundError("Missing opened_hubs.csv. Run: python project.py optimize")
    if not assigns_path.exists():
        raise FileNotFoundError("Missing assignments.csv. Run: python project.py optimize")

    cafes     = pd.read_csv(cafes_path)
    hubs_open = pd.read_csv(hubs_open_path)
    assigns   = pd.read_csv(assigns_path)
    flows = pd.read_csv(flows_path) if flows_path.exists() else pd.DataFrame()

    dcs_open = pd.read_csv(dcs_open_path) if dcs_open_path.exists() else pd.DataFrame()

    print("[DEBUG] cafes:", cafes.shape, "hubs_open:", hubs_open.shape, "dcs_open:", dcs_open.shape, "assigns:", assigns.shape)
    print("[DEBUG] assigns columns:", list(assigns.columns))
    print("[DEBUG] hubs_open columns:", list(hubs_open.columns))

    center_lat = float(pd.to_numeric(cafes["lat"], errors="coerce").median())
    center_lon = float(pd.to_numeric(cafes["lon"], errors="coerce").median())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles=None)

    folium.TileLayer("CartoDB positron", name="Carto (clean)", control=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    MeasureControl(position="topright", primary_length_unit="kilometers").add_to(m)
    LocateControl(auto_start=False).add_to(m)

    fg_dc_to_hub   = folium.FeatureGroup(name="Primary Supply (DC → Hub)", show=True).add_to(m)
    fg_hub_to_cafe = folium.FeatureGroup(name="Secondary Delivery (Hub → Cafe)", show=False).add_to(m)
    fg_nodes       = folium.FeatureGroup(name="Nodes (DC, Hub, Cafe)", show=True).add_to(m)

    need_cols = ["lat_cafe", "lon_cafe", "lat_hub", "lon_hub", "dist_km"]
    if not all(c in assigns.columns for c in need_cols):
        cafes_xy = cafes[["lat","lon","city","country"]].copy().rename(columns={
            "lat":"lat_cafe","lon":"lon_cafe",
            "city":"city_cafe","country":"country_cafe"
        })
        cafes_xy["cafe_idx"] = cafes_xy.index

        if "hub_idx" in hubs_open.columns:
            hubs_xy = hubs_open[["hub_idx","lat","lon","city","country"]].copy().rename(columns={
                "lat":"lat_hub","lon":"lon_hub","city":"city_hub","country":"country_hub"
            })
            assigns = assigns.merge(cafes_xy, on="cafe_idx", how="left")
            assigns = assigns.merge(hubs_xy, on="hub_idx", how="left")
        else:
            assigns = assigns.merge(cafes_xy, on="cafe_idx", how="left")

    for c in ["lat_cafe","lon_cafe","lat_hub","lon_hub","dist_km"]:
        if c in assigns.columns:
            assigns[c] = pd.to_numeric(assigns[c], errors="coerce")

    assigns_clean = assigns.dropna(subset=[c for c in need_cols if c in assigns.columns]).copy()
    print("[DEBUG] assigns_clean:", assigns_clean.shape)

    if "hub_idx" not in hubs_open.columns:
        recovered = False

        if candidates_path.exists() and ("geonameid" in hubs_open.columns):
            cand = pd.read_csv(candidates_path)
            if "geonameid" in cand.columns:
                geonameid_to_idx = {int(g): int(i) for i, g in enumerate(cand["geonameid"].astype(int).tolist())}
                hubs_open["hub_idx"] = hubs_open["geonameid"].astype(int).map(geonameid_to_idx)
                if (not dcs_open.empty) and ("geonameid" in dcs_open.columns):
                    dcs_open["hub_idx"] = dcs_open["geonameid"].astype(int).map(geonameid_to_idx)

                if hubs_open["hub_idx"].notna().any():
                    recovered = True
                    print("[INFO] hub_idx recovered via candidates_50.csv (geonameid -> row index).")

        if (not recovered) and all(c in assigns_clean.columns for c in ["hub_idx","city_hub","country_hub","lat_hub","lon_hub"]):
            hub_key = assigns_clean[["hub_idx","city_hub","country_hub","lat_hub","lon_hub"]].drop_duplicates().copy()
            hub_key["lat_hub"] = pd.to_numeric(hub_key["lat_hub"], errors="coerce")
            hub_key["lon_hub"] = pd.to_numeric(hub_key["lon_hub"], errors="coerce")

            hubs_open2 = hubs_open.copy()
            if all(c in hubs_open2.columns for c in ["city","country","lat","lon"]):
                hubs_open2["lat"] = pd.to_numeric(hubs_open2["lat"], errors="coerce")
                hubs_open2["lon"] = pd.to_numeric(hubs_open2["lon"], errors="coerce")

                hubs_open = hubs_open.merge(
                    hub_key.rename(columns={"city_hub":"city","country_hub":"country","lat_hub":"lat","lon_hub":"lon"}),
                    on=["city","country","lat","lon"],
                    how="left"
                )
                if "hub_idx_y" in hubs_open.columns and "hub_idx_x" in hubs_open.columns:
                    hubs_open["hub_idx"] = hubs_open["hub_idx_x"].fillna(hubs_open["hub_idx_y"])
                    hubs_open = hubs_open.drop(columns=[c for c in ["hub_idx_x","hub_idx_y"] if c in hubs_open.columns])

                if not dcs_open.empty and all(c in dcs_open.columns for c in ["city","country","lat","lon"]):
                    dcs_open["lat"] = pd.to_numeric(dcs_open["lat"], errors="coerce")
                    dcs_open["lon"] = pd.to_numeric(dcs_open["lon"], errors="coerce")
                    dcs_open = dcs_open.merge(
                        hub_key.rename(columns={"city_hub":"city","country_hub":"country","lat_hub":"lat","lon_hub":"lon"}),
                        on=["city","country","lat","lon"],
                        how="left"
                    )
                if hubs_open.get("hub_idx").notna().any():
                    recovered = True
                    print("[INFO] hub_idx recovered via assignments (city/country/lat/lon match).")

        if "hub_idx" not in hubs_open.columns or hubs_open["hub_idx"].isna().all():
            hubs_open["hub_idx"] = range(len(hubs_open))
            if not dcs_open.empty and "hub_idx" not in dcs_open.columns:
                dcs_open["hub_idx"] = range(len(dcs_open))
            print("[WARN] hub_idx could not be matched reliably. Falling back to row index (map still works).")

    if "hub_idx" in assigns_clean.columns:
        hub_stats = (
            assigns_clean.groupby("hub_idx")
                .agg(num_cafes=("cafe_idx", "count"),
                     avg_dist=("dist_km", "mean"),
                     max_dist=("dist_km", "max"))
                .reset_index()
        )
        hub_stats_map = hub_stats.set_index("hub_idx").to_dict(orient="index")
    else:
        hub_stats_map = {}

    if all(c in assigns_clean.columns for c in ["lat_cafe","lon_cafe","lat_hub","lon_hub"]):
        for _, row in assigns_clean.iterrows():
            folium.PolyLine(
                locations=[[row["lat_cafe"], row["lon_cafe"]], [row["lat_hub"], row["lon_hub"]]],
                weight=3,          
                color="#2c3e50",   
                opacity=0.90,      
                
                tooltip=f"Hub→Cafe | {row.get('city_hub','?')}→{row.get('city_cafe','?')} | {float(row['dist_km']):.1f} km",
            ).add_to(fg_hub_to_cafe)


    if (dcs_open is not None) and (not dcs_open.empty) and (flows is not None) and (not flows.empty):
        dcs_xy = dcs_open[["hub_idx","city","country","lat","lon"]].copy()
        dcs_xy["lat"] = pd.to_numeric(dcs_xy["lat"], errors="coerce")
        dcs_xy["lon"] = pd.to_numeric(dcs_xy["lon"], errors="coerce")
        dcs_xy = dcs_xy.dropna(subset=["lat","lon"]).set_index("hub_idx")

        hubs_xy = hubs_open[["hub_idx","city","country","lat","lon"]].copy()
        hubs_xy["lat"] = pd.to_numeric(hubs_xy["lat"], errors="coerce")
        hubs_xy["lon"] = pd.to_numeric(hubs_xy["lon"], errors="coerce")
        hubs_xy = hubs_xy.dropna(subset=["lat","lon"]).set_index("hub_idx")

        flows2 = flows.copy()
        flows2["ton"] = pd.to_numeric(flows2["ton"], errors="coerce").fillna(0.0)

        agg = (flows2.groupby(["dc_idx","hub_idx"], as_index=False)
                    .agg(ton=("ton","sum")))

        for row in agg.itertuples(index=False):
            dc_idx = int(row.dc_idx)
            hub_idx = int(row.hub_idx)
            ton = float(row.ton)

            if dc_idx not in dcs_xy.index or hub_idx not in hubs_xy.index:
                continue

            d = dcs_xy.loc[dc_idx]
            h = hubs_xy.loc[hub_idx]

            w = 2.0 + min(10.0, ton / 40.0)

            folium.PolyLine(
                locations=[[float(d.lat), float(d.lon)], [float(h.lat), float(h.lon)]],
                weight=float(w),
                color="#2c3e50",
                opacity=0.90,
                tooltip=f"MILP DC→Hub | {d.city}→{h.city} | ton={ton:.1f}",
            ).add_to(fg_dc_to_hub)
    else:
        print("[WARN] No dc_hub_flows.csv found (or empty). Run optimize again to export true MILP flows.")

    cafe_cluster = MarkerCluster(name="Cafes Cluster").add_to(fg_nodes)
    for _, r in cafes.iterrows():
        lat = pd.to_numeric(r.get("lat"), errors="coerce")
        lon = pd.to_numeric(r.get("lon"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        folium.CircleMarker(
            [float(lat), float(lon)],
            radius=3,
            color="#666666",
            weight=1,
            fill=True,
            fill_color="#aaaaaa",
            fill_opacity=0.75,
            tooltip=f"CAFE: {r.get('city','?')} ({r.get('country','?')})",
        ).add_to(cafe_cluster)

    for _, r in hubs_open.iterrows():
        lat = pd.to_numeric(r.get("lat"), errors="coerce")
        lon = pd.to_numeric(r.get("lon"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue

        country = str(r.get("country", ""))
        hub_idx = int(r.get("hub_idx")) if pd.notna(r.get("hub_idx")) else None

        stats = hub_stats_map.get(hub_idx, {}) if hub_idx is not None else {}
        usage = int(stats.get("num_cafes", 0))
        avg_d = float(stats.get("avg_dist", float("nan"))) if stats else float("nan")
        mx_d  = float(stats.get("max_dist", float("nan"))) if stats else float("nan")

        radius = 7 + min(11, usage / 6.0)
        color = "#ff7a00" if country == "TR" else "#1f77ff"

        tip = (
            f"HUB: {r.get('city','?')} ({country})"
            f" | cafes={usage}"
            + (f" | avg={avg_d:.1f} km" if not math.isnan(avg_d) else "")
            + (f" | max={mx_d:.1f} km" if not math.isnan(mx_d) else "")
        )

        folium.CircleMarker(
            [float(lat), float(lon)],
            radius=float(radius),
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.90,
            tooltip=tip,
        ).add_to(fg_nodes)

    if dcs_open is not None and not dcs_open.empty:
        for _, r in dcs_open.iterrows():
            lat = pd.to_numeric(r.get("lat"), errors="coerce")
            lon = pd.to_numeric(r.get("lon"), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue
            folium.CircleMarker(
                [float(lat), float(lon)],
                radius=12,
                color="#cc0000",
                weight=3,
                fill=True,
                fill_color="#ff0000",
                fill_opacity=0.95,
                tooltip=f"DC: {r.get('city','?')} ({r.get('country','?')})",
            ).add_to(fg_nodes)

    legend_html = """
    <div style="
    position: fixed; bottom: 20px; left: 20px; width: 280px; z-index:9999;
    background: white; padding: 12px; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    font-size: 12px;">
    <b>Legend</b><br>
    <span style="color:#cc0000;">●</span> DC (big)<br>
    <span style="color:#1f77ff;">●</span> Hub EU (medium, sized by utilization)<br>
    <span style="color:#ff7a00;">●</span> Hub TR (medium, sized by utilization)<br>
    <span style="color:#888888;">●</span> Cafe (small, clustered)<br>
    <hr style="margin:8px 0">
    <span style="color:#2c3e50;">━</span> Primary Supply (DC→Hub)<br>
    <span style="color:#777777;">- -</span> Secondary Delivery (Hub→Cafe)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out_path = OUTPUTS / "professional_supply_chain_map.html"
    m.save(str(out_path))
    print("Wrote:", out_path)


def step_scenarios():
    ensure_dirs()
    base_cfg = load_config()

    sla_list = [1400, 1600, 1800]
    fixed_list = [1000, 2000, 4000]
    one_dc_list = [False, True]

    rows = []
    for sla in sla_list:
        for fc in fixed_list:
            for one_dc in one_dc_list:
                cfg = copy.deepcopy(base_cfg)
                cfg["max_service_km"] = sla
                cfg["fixed_cost_scale"] = fc
                cfg["one_dc_per_product"] = one_dc
                save_config(cfg)
                print(f"=== Scenario: SLA={sla}, fixed_cost_scale={fc}, one_dc={one_dc} ===")

                try:
                    step_optimize()
                    summ = pd.read_csv(OUTPUTS / "solution_summary.csv").iloc[0].to_dict()
                    summ["scenario_sla"] = sla
                    summ["scenario_fc"] = fc
                    summ["scenario_one_dc"] = one_dc
                except RuntimeError as e:
                    summ = {
                        "status": "Infeasible",
                        "error": str(e),
                        "scenario_sla": sla,
                        "scenario_fc": fc,
                        "scenario_one_dc": one_dc,
                    }

                rows.append(summ)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUTS / "scenario_summary.csv", index=False, encoding="utf-8")
    print("Wrote:", OUTPUTS / "scenario_summary.csv")

    save_config(base_cfg)


def step_all():
    step_top100()
    step_poi()
    step_candidates()
    step_optimize()
    step_map()
    step_scenarios()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("top100")
    sub.add_parser("poi")
    sub.add_parser("candidates")
    sub.add_parser("optimize")
    sub.add_parser("map")
    sub.add_parser("scenarios")
    sub.add_parser("all")

    args = parser.parse_args()
    if not args.cmd:
        args.cmd = "all"

    if args.cmd == "top100":
        step_top100()
    elif args.cmd == "poi":
        step_poi()
    elif args.cmd == "candidates":
        step_candidates()
    elif args.cmd == "optimize":
        step_optimize()
    elif args.cmd == "map":
        step_map()
    elif args.cmd == "scenarios":
        step_scenarios()
    elif args.cmd == "all":
        step_all()


if __name__ == "__main__":
    main()
