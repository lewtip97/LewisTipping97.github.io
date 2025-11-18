from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import LineString


# Data path - works both locally and on Render
# On Render, the data directory should be in the repo root
DATA_PATH = Path(__file__).parent.parent / "data" / "prow" / "ldnpa_prow.gpkg"
MAX_GRAPH_EDGES = 5_000_000
NODE_PRECISION = 3  # metres


class Fell(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    summited: bool


class HikeRequest(BaseModel):
    max_distance_km: float = Field(gt=0, description="Maximum length for each hike in kilometres.")
    fells: List[Fell]

    @validator("fells")
    def ensure_fells(cls, value: List[Fell]) -> List[Fell]:
        if not value:
            raise ValueError("No fells provided.")
        return value


class RouteSegment(BaseModel):
    coordinates: List[Tuple[float, float]]


class HikeResponse(BaseModel):
    id: str
    fell_ids: List[str]
    fell_names: List[str]
    distance_km: float
    geometry: RouteSegment


class HikeCollection(BaseModel):
    hikes: List[HikeResponse]


app = FastAPI(title="Wainwright Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GraphStore:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Routing dataset not found at {path}")

        self.graph = nx.Graph()
        self.transform_to_wgs84 = Transformer.from_crs(27700, 4326, always_xy=True)
        self.offset_nodes: np.ndarray | None = None
        self.node_ids: List[int] = []

        self._load_graph(path)

    def _load_graph(self, path: Path) -> None:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError("Routing dataset has no CRS. Please ensure it is defined (EPSG:27700).")
        gdf = gdf.to_crs(epsg=27700)

        nodes: Dict[Tuple[int, int], int] = {}

        def add_node(x: float, y: float) -> int:
            key = (int(round(x, NODE_PRECISION)), int(round(y, NODE_PRECISION)))
            node_id = nodes.get(key)
            if node_id is not None:
                return node_id

            node_id = len(nodes)
            nodes[key] = node_id
            self.graph.add_node(node_id, x=x, y=y)
            return node_id

        edge_count = 0
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, LineString):
                lines = [geom]
            else:
                lines = list(geom.geoms)

            for line in lines:
                coords = list(line.coords)
                for start, end in zip(coords[:-1], coords[1:]):
                    u = add_node(*start)
                    v = add_node(*end)
                    length = LineString([start, end]).length
                    if length == 0:
                        continue

                    if self.graph.has_edge(u, v):
                        existing = self.graph[u][v]["length"]
                        if length < existing:
                            self.graph[u][v]["length"] = length
                            self.graph[u][v]["geometry"] = LineString([start, end])
                    else:
                        self.graph.add_edge(
                            u,
                            v,
                            length=length,
                            geometry=LineString([start, end]),
                        )
                        edge_count += 1

        if edge_count == 0:
            raise ValueError("No edges were created from the routing dataset.")
        if edge_count > MAX_GRAPH_EDGES:
            raise ValueError("Graph appears unreasonably large. Please verify the dataset.")

        self.node_ids = list(self.graph.nodes())
        coords = [(self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in self.node_ids]
        self.offset_nodes = np.array(coords, dtype=np.float64)

    def nearest_node(self, x: float, y: float) -> int:
        if self.offset_nodes is None:
            raise RuntimeError("Graph nodes have not been initialised.")

        point = np.array([x, y], dtype=np.float64)
        distances = np.square(self.offset_nodes - point).sum(axis=1)
        idx = int(np.argmin(distances))
        return self.node_ids[idx]

    @lru_cache(maxsize=10_000)
    def shortest_path_length(self, u: int, v: int) -> float:
        return nx.shortest_path_length(self.graph, u, v, weight="length")

    @lru_cache(maxsize=10_000)
    def shortest_path_nodes(self, u: int, v: int) -> Tuple[int, ...]:
        return tuple(nx.shortest_path(self.graph, u, v, weight="length"))

    def nodes_to_linestring(self, path: Tuple[int, ...]) -> LineString:
        coords = [(self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in path]
        return LineString(coords)

    def project_linestring(self, line: LineString) -> LineString:
        lon, lat = self.transform_to_wgs84.transform(*line.xy)
        return LineString(zip(lon, lat))


graph_store: GraphStore | None = None


@app.on_event("startup")
def initialise_graph() -> None:
    global graph_store
    graph_store = GraphStore(DATA_PATH)


def group_fells(
    fell_indices: List[int],
    node_ids: List[int],
    max_length: float,
    distance_cache: Dict[Tuple[int, int], float],
) -> List[List[int]]:
    remaining = set(fell_indices)
    groups: List[List[int]] = []

    while remaining:
        current = remaining.pop()
        group = [current]
        cumulative = 0.0

        while remaining:
            best_idx = None
            best_increase = None

            for candidate in list(remaining):
                key = tuple(sorted((group[-1], candidate)))
                if key in distance_cache:
                    dist = distance_cache[key]
                else:
                    try:
                        dist = graph_store.shortest_path_length(
                            node_ids[group[-1]], node_ids[candidate]
                        )
                    except nx.NetworkXNoPath:
                        continue
                    distance_cache[key] = dist

                tentative = cumulative + dist
                if tentative <= max_length and (best_increase is None or dist < best_increase):
                    best_idx = candidate
                    best_increase = dist

            if best_idx is None:
                break

            remaining.remove(best_idx)
            group.append(best_idx)
            cumulative += best_increase or 0.0

        groups.append(group)

    return groups


@app.get("/")
def root():
    return {"message": "Wainwright Planner API", "endpoints": {"POST /api/hikes": "Generate hike routes"}}

@app.post("/api/hikes", response_model=HikeCollection)
def create_hikes(payload: HikeRequest) -> HikeCollection:
    if graph_store is None:
        raise HTTPException(status_code=503, detail="Routing graph not ready.")

    max_distance_m = payload.max_distance_km * 1000
    if max_distance_m <= 0:
        raise HTTPException(status_code=400, detail="Maximum distance must be positive.")

    unsummited = [fell for fell in payload.fells if not fell.summited]
    if not unsummited:
        return HikeCollection(hikes=[])

    # Build numpy arrays for coordinate transforms
    transformer = Transformer.from_crs(4326, 27700, always_xy=True)
    fell_coords = np.array(
        [transformer.transform(fell.lon, fell.lat) for fell in unsummited], dtype=np.float64
    )
    fell_node_ids = [graph_store.nearest_node(x, y) for x, y in fell_coords]

    distance_cache: Dict[Tuple[int, int], float] = {}
    idxs = list(range(len(unsummited)))
    grouped = group_fells(idxs, fell_node_ids, max_distance_m, distance_cache)

    hikes: List[HikeResponse] = []
    for group_idx, group in enumerate(grouped, start=1):
        if len(group) == 1:
            node = graph_store.graph.nodes[fell_node_ids[group[0]]]
            line = LineString([(node["x"], node["y"]), (node["x"], node["y"])])
            projected = graph_store.project_linestring(line)
            total_length = 0.0
        else:
            ordered_nodes: List[int] = []
            total_length = 0.0
            for i in range(len(group) - 1):
                u_idx, v_idx = group[i], group[i + 1]
                path_nodes = graph_store.shortest_path_nodes(fell_node_ids[u_idx], fell_node_ids[v_idx])
                if i == 0:
                    ordered_nodes.extend(path_nodes)
                else:
                    ordered_nodes.extend(path_nodes[1:])
                total_length += graph_store.shortest_path_length(fell_node_ids[u_idx], fell_node_ids[v_idx])

            line = graph_store.nodes_to_linestring(tuple(ordered_nodes))
            projected = graph_store.project_linestring(line)

        hike = HikeResponse(
            id=f"hike-{group_idx}",
            fell_ids=[unsummited[i].id for i in group],
            fell_names=[unsummited[i].name for i in group],
            distance_km=round(total_length / 1000, 2),
            geometry=RouteSegment(coordinates=list(projected.coords)),
        )
        hikes.append(hike)

    return HikeCollection(hikes=hikes)

