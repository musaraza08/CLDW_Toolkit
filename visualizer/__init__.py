from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


METHOD_COLORS = {
    "LLM": "red",
    "NLP": "blue",
    "AUTO": "green",
    "GT": "darkgreen",
}


class TrajectoryVisualizer:
    def __init__(self, user_agent: str = "travel_trajectory_visualizer", manual_gt_path: Path = Path("results/manual_trajectories.json")) -> None:
        self.geolocator = Nominatim(user_agent=user_agent)
        self.manual_gt_path = Path(manual_gt_path)
        self._manual_index: Dict[str, Dict] = {}
        self._load_manual()

    def _load_manual(self) -> None:
        try:
            import json
            if self.manual_gt_path.exists():
                with self.manual_gt_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                for rec in data:
                    rid = rec.get("id") or rec.get("ID")
                    if rid:
                        self._manual_index[str(rid)] = rec
        except Exception:
            self._manual_index = {}

    def _geocode_place(self, place_name: str, max_retries: int = 3) -> Optional[Tuple[float, float]]:
        for attempt in range(max_retries):
            try:
                # geocoding for British places
                search_query = f"{place_name}, England, UK"
                location = self.geolocator.geocode(search_query, timeout=10)
                if location:
                    return location.latitude, location.longitude

                # Fallback to raw query
                location = self.geolocator.geocode(place_name, timeout=10)
                if location:
                    return location.latitude, location.longitude
            except (GeocoderTimedOut, GeocoderUnavailable):
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            except Exception:
                return None
        return None

    @staticmethod
    def _extract_method_trajectories(record: Dict) -> List[Tuple[str, Dict]]:
        out: List[Tuple[str, Dict]] = []
        if isinstance(record, dict):
            if "llm_result" in record or "nlp_result" in record:
                if record.get("llm_result") and record["llm_result"].get("trajectory"):
                    out.append(("LLM", record["llm_result"]))
                if record.get("nlp_result") and record["nlp_result"].get("trajectory"):
                    out.append(("NLP", record["nlp_result"]))
            elif record.get("trajectory"):
                label = record.get("method") or "AUTO"
                out.append((label, record))
        return out

    def _extract_gt_trajectory(self, record: Dict) -> Optional[Dict]:
        rec_id = record.get("ID") or record.get("id")
        if not rec_id:
            return None
        manual = self._manual_index.get(str(rec_id))
        if not manual or not manual.get("trajectory"):
            return None
        # Package to look like other method result blocks
        return {
            "trajectory": manual.get("trajectory", []),
            "Author": record.get("Author"),
            "Year": record.get("Year"),
            "method": "GT",
        }

    def _build_map_from_layers(self, record: Dict, layers: List[Tuple[str, Dict]]) -> Optional[folium.Map]:
        if not layers:
            return None

        m = folium.Map(location=[54.0, -2.0],
                       zoom_start=6, tiles="OpenStreetMap")

        for method_label, result in layers:
            traj = result.get("trajectory") or []
            if not traj:
                continue

            color = METHOD_COLORS.get(method_label, "purple")
            author = result.get("Author") or record.get("Author")
            year = result.get("Year") or record.get("Year")

            fg_name = f"{method_label} – {author} ({year})" if author or year else f"{method_label}"
            fg = folium.FeatureGroup(name=fg_name)

            coordinates: List[List[float]] = []
            for place_info in traj:
                # place_info may be a string in manual GT
                if isinstance(place_info, str):
                    place_name = place_info
                    lat = None
                    lon = None
                else:
                    place_name = place_info.get("place")
                    lat = place_info.get("lat")
                    lon = place_info.get("lon") or place_info.get("long")

                coords: Optional[Tuple[float, float]]
                if lat is not None and lon is not None:
                    try:
                        coords = (float(lat), float(lon))
                    except Exception:
                        coords = None
                else:
                    coords = self._geocode_place(
                        place_name) if place_name else None

                if coords:
                    lat_v, lon_v = coords
                    coordinates.append([lat_v, lon_v])

                    popup_text = f"""
                    <b>{place_name}</b><br>
                    Order: {place_info.get('order','') if isinstance(place_info, dict) else ''}<br>
                    Context: {place_info.get('context','') if isinstance(place_info, dict) else ''}<br>
                    Method: {method_label}<br>
                    Author: {author or ''}<br>
                    Year: {year or ''}
                    """
                    folium.Marker(
                        [lat_v, lon_v],
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=folium.Icon(color=color, icon="info-sign")
                    ).add_to(fg)

            if len(coordinates) > 1:
                folium.PolyLine(
                    coordinates,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Route: {method_label} – {author} ({year})" if author or year else f"Route: {method_label}"
                ).add_to(fg)

            fg.add_to(m)

        folium.LayerControl().add_to(m)
        return m

    def build_record_map(self, record: Dict) -> Optional[folium.Map]:
        methods = self._extract_method_trajectories(record)
        return self._build_map_from_layers(record, methods)

    def build_combined_map(self, record: Dict) -> Optional[folium.Map]:
        layers = []
        gt = self._extract_gt_trajectory(record)
        if gt:
            layers.append(("GT", gt))
        layers.extend(self._extract_method_trajectories(record))
        return self._build_map_from_layers(record, layers)

    def build_gt_vs_llm_map(self, record: Dict) -> Optional[folium.Map]:
        layers: List[Tuple[str, Dict]] = []
        gt = self._extract_gt_trajectory(record)
        if gt:
            layers.append(("GT", gt))
        if isinstance(record.get("llm_result"), dict):
            llm = record["llm_result"]
            if llm.get("trajectory"):
                layers.append(("LLM", llm))
        return self._build_map_from_layers(record, layers)

    def build_gt_vs_nlp_map(self, record: Dict) -> Optional[folium.Map]:
        layers: List[Tuple[str, Dict]] = []
        gt = self._extract_gt_trajectory(record)
        if gt:
            layers.append(("GT", gt))
        if isinstance(record.get("nlp_result"), dict):
            nlp = record["nlp_result"]
            if nlp.get("trajectory"):
                layers.append(("NLP", nlp))
        return self._build_map_from_layers(record, layers)

    def save_record_map(self, record: Dict, output_html: Path) -> Optional[Path]:
        m = self.build_record_map(record)
        if m is None:
            return None
        output_html.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_html))
        return output_html

    def save_map(self, m: Optional[folium.Map], output_html: Path) -> Optional[Path]:
        if m is None:
            return None
        output_html.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_html))
        return output_html

    def save_all_maps_for_record(self, record: Dict, out_dir: Path) -> Dict[str, Optional[Path]]:

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Optional[Path]] = {}
        saved["gt_vs_llm"] = self.save_map(
            self.build_gt_vs_llm_map(record), out_dir / "map_gt_vs_llm.html")
        saved["gt_vs_nlp"] = self.save_map(
            self.build_gt_vs_nlp_map(record), out_dir / "map_gt_vs_nlp.html")
        saved["combined"] = self.save_map(
            self.build_combined_map(record), out_dir / "map_combined.html")
        return saved


__all__ = ["TrajectoryVisualizer"]
