# Facility Location & Multi-Echelon Network Design (Europe + Turkey) — MILP Decision Support

This repository presents an end-to-end **multi-echelon facility location & network design** decision-support project for a hypothetical **Study-Café** chain operating across **Europe and Turkey**.

The project answers a strategic supply chain question:

> **Where should we open cross-dock hubs and regional distribution centers (DCs), and how should we allocate demand cities to hubs, while managing the trade-off between total network cost and service feasibility?**

The pipeline integrates **real geospatial data**, **cost proxies**, and a **Mixed-Integer Linear Programming (MILP)** formulation and delivers results as:
- structured CSV decision outputs, and
- an interactive **Folium** network map.

---

## Quick Results (Baseline)

- **Status:** Optimal  
- **Opened facilities:** **12 hubs**, **2 DCs**  
- **Average café-to-hub distance:** ~334.6 km  
- **Max café-to-hub distance:** ~1121.8 km (within **1400 km** SLA)  
- **Objective value:** ~38954.76  
  - hub fixed: ~14654.74  
  - dc fixed: ~14061.60  
  - inbound: ~3496.27  
  - outbound: ~6742.15  

---

## 1) Business Context & Decision Scope

### Network structure (multi-echelon)
- **DC → Hub → Demand Cities (Cafés)**
- **DCs** act as upstream consolidation points (regional distribution).
- **Hubs** act as cross-dock / transshipment points to improve service coverage and reduce last-mile distance exposure.

### Strategic decisions the model makes
- Which candidate cities should be opened as **hubs** (binary).
- Which candidate cities should be opened as **DCs** (binary).
- Which **hub** each demand city is assigned to (binary assignment).
- How product flows move from **DCs to hubs** (continuous flows) to support hub–city demand.

### What “service” means in this project
Service is modeled through a **distance-based SLA**:
- a demand city may only be assigned to a hub if distance ≤ `max_service_km`.
- distance acts as a practical proxy for lead-time and responsiveness at strategic level.

---

## 2) Key Drivers for Facility Site Selection (From Scratch)

Facility choice is driven by demand potential, infrastructure attractiveness, operating economics, and feasibility constraints.

### 2.1 Demand potential (market attractiveness)
Demand is represented using a **population-based proxy** and scaled by a configurable multiplier:

- Population is normalized and combined with a baseline factor:
  - `base_weight = (0.3 + 0.7 * pop_norm) * demand_factor * demand_multiplier`
- `demand_factor` is derived from POI signals.
- This reflects the assumption that larger and more active cities generate higher café demand.

### 2.2 POI-derived activity & infrastructure signals (OpenStreetMap)
The project uses **OpenStreetMap (Overpass API)** to compute POI counts for each city within a configurable radius (default **10 km**):

- **Student intensity:** university / college / school  
- **Activity intensity:** mall / cinema / café  
- **Logistics intensity:** logistics / warehouse  
- **Transit intensity:** station / bus station  

These signals are used to construct:
- a **demand factor** (student + activity emphasis), and
- a **hub attractiveness score** (logistics + transit emphasis).

To reduce external API calls, POI responses are cached locally:
- `data_out/poi_cache/*.json`

### 2.3 Operating economics (fixed cost proxy: rent + labour + tax)
A key objective of the project is to capture the strategic trade-off:

> **Opening more facilities improves service coverage but increases fixed operating cost.**

Fixed operating cost is constructed using scalable proxies:
- **Rent** (city-level if available; else country mean; else global mean)
- **Labour cost index** (country-level)
- **Corporate income tax (CIT)** (country-level)

A weighted multiplier is computed:

`fixed_mult = 1 + w_r * rent_n + w_l * labour_n + w_t * tax_n`

Then:
- `fixed_cost_hub = base_hub_cost * fixed_mult`
- `fixed_cost_dc  = base_dc_cost  * fixed_mult`

Default weights (configurable):
- rent `0.45`, labour `0.45`, tax `0.10`

**Missing-data handling**
- Rent resolves via **city → country → global** fallback.
- The source used is recorded per candidate (e.g., `rent_source_used`).
- Warnings are logged for transparency.

### 2.4 Infrastructure discount on fixed costs
To reflect operational advantages of well-connected cities, an infrastructure-based discount is applied:

- `fixed_cost_hub_eff = fixed_cost_hub * (1 - disc_hub * hub_score_n)`
- `fixed_cost_dc_eff  = fixed_cost_dc  * (1 - disc_dc  * hub_score_n)`
- Discount is bounded so effective fixed cost does not fall below **70%** of the original.

### 2.5 Service feasibility (SLA constraint)
A demand city can only be served by a hub if:
- `distance(city, hub) ≤ max_service_km`

Demand cities with no feasible hub are reported in:
- `outputs/uncovered_cafes.csv`

### 2.6 Network coherence and regional resilience
The model enforces:
- logical linking between facility opening and usage (no assignment or flow through closed sites),
- capacity-style limits (tonnage and café count),
- cross-border resilience by enforcing **both TR and EU DC presence**.

---

## 3) Candidate Facility Pool Generation (Hub Candidates)

Instead of allowing every city to become a hub, a controlled candidate pool is generated.

### 3.1 Candidate scoring
Each city receives a candidate score combining:
- infrastructure attractiveness (`hub_score_n`)
- demand attractiveness (`demand_n`)
- operating cost penalty (`fixed_mult_n`)

`candidate_score = w_infra * hub_score_n + w_demand * demand_n - w_cost * fixed_mult_n`

Default weights:
- infrastructure `0.55`
- demand `0.15`
- cost `0.30`

### 3.2 Spatial thinning and country caps
To avoid spatial redundancy:
- **Spatial thinning** is applied (default **30 km**).
- **Maximum candidates per country** is enforced (default **20**).
- Region minimums ensure both **TR** and **EU** representation.

Outputs:
- `data_out/candidates_50.csv`
- `data_out/candidate_pool_diagnostics.csv`

---

## 4) Products & Cost Structure (Multi-Commodity)

The network supports multiple product categories (example set):
- `beans`
- `syrup`
- `cons`

For each product:
- inbound cost (DC → hub) is modeled by `in_cost_per_ton_km`,
- outbound cost (hub → city) is modeled by `out_cost_per_ton_km`,
- demand per city is derived via `base_ton_per_city × demand_weight`.

This allows different products to share the same network while retaining different transport economics.

---

## 5) Optimization Model (MILP)

### 5.1 Decision variables
- `y_h[j]` = 1 if hub `j` is opened
- `y_d[k]` = 1 if DC `k` is opened
- `x[i,j]` = 1 if demand city `i` is assigned to hub `j`
- `f[k,j,p]` = flow of product `p` from DC `k` to hub `j`

### 5.2 Objective function
Minimize total system cost:
- fixed facility costs (hubs and DCs),
- inbound transport (DC → hub),
- outbound transport (hub → demand),
- optional CO₂ cost (disabled in baseline).

### 5.3 Core constraints
- service feasibility (SLA),
- single-hub assignment per demand city,
- facility linking (open-to-use),
- minimum and maximum number of hubs and DCs,
- capacity-style constraints,
- regional enforcement (TR + EU DC).

Solver:
- **PuLP (CBC)**

---

## 6) Baseline Configuration (from `config.yaml`)

- `max_service_km = 1400`
- `min_hubs = 12`, `max_hubs = 50`
- `candidate_hubs_n = 50`
- `demand_multiplier = 2.5`
- `poi_radius_m = 10000`

Fixed cost:
- `fixed_cost_scale = 1000`
- `base_hub_cost = 1000`
- `base_dc_cost = 6000`

Capacity controls:
- hub max cafés = `60`
- hub max ton = `120`
- dc max ton = `400`

Region:
- TR + EU DC enforcement enabled

---

## 7) How to Run

### 7.1 Install
```bash
pip install -r requirements.txt

### 7.2 Prepare required raw inputs (data_raw/)
Required files:
GeoNames

cities15000.txt or cities15000.zip
countryInfo.txt
Cost proxies

labour_cost_country.csv
cit_rate_country.csv
rent_city.csv (optional)
### 7.3 Run steps

python project.py top100
python project.py poi
python project.py candidates
python project.py optimize
python project.py map
Run everything:


python project.py all
Scenario analysis:


python project.py scenarios
## 8) Outputs
Data preparation (data_out/)
top100_cities.csv
poi_scores_100.csv
candidates_50.csv
candidate_pool_diagnostics.csv
poi_cache/*.json
Optimization outputs (outputs/)
solution_summary.csv
opened_hubs.csv
opened_dcs.csv
assignments.csv
dc_hub_flows.csv
candidate_costs.csv
uncovered_cafes.csv
scenario_summary.csv
professional_supply_chain_map.html
Open the map:

outputs/professional_supply_chain_map.html
## 9) Interpretation Guide
opened_hubs.csv / opened_dcs.csv: selected facilities
assignments.csv: demand city → hub mapping
dc_hub_flows.csv: upstream consolidation
solution_summary.csv: KPIs and objective breakdown
Insights typically come from SLA tightening, cost scaling, and regional enforcement comparisons.
## 10) Limitations & Extensions
This is a strategic-level model:

SLA is distance-based.
Capacities are stylized.
Costs rely on proxies.
Possible extensions:

time-based SLA,
stochastic demand,
multi-period planning,
CO₂ trade-off analysis,
facility capacity investment.
11) Data Attribution
GeoNames
OpenStreetMap (Overpass API)
Please include proper attribution when redistributing derived outputs.
Author
Mehmet Ali Hoca

Industrial Engineering — Supply Chain Optimization & Decision Support Systems
