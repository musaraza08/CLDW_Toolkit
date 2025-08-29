from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from lxml import etree as ET

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

METADATA_CSV = Path("data/LakeDistrictCorpus/LD80_metadata/LD80_metadata.csv")
GEOPARSED_DIR = Path("data/LakeDistrictCorpus/LD80_geoparsed")

# ---------------------------------------------------------------------------
# DataLoader class
# ---------------------------------------------------------------------------


class DataLoader:
    """Load CLDW metadata and serve geoparsed records & plain text."""

    def __init__(self, metadata_csv: Path = METADATA_CSV, geoparsed_dir: Path = GEOPARSED_DIR):
        self.metadata_csv = metadata_csv
        self.geoparsed_dir = geoparsed_dir
        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {self.metadata_csv}")
        self.df = pd.read_csv(self.metadata_csv)
        # Filter to include only specific genres
        allowed_genres = {"Travelogue", "Journal",
                          "Guide", "Survey", "Epistle"}
        # Ensure Genre column exists and filter rows
        if "Genre" in self.df.columns:
            self.df = self.df[self.df["Genre"].isin(allowed_genres)].copy()
        else:
            raise KeyError("'Genre' column not found in metadata CSV.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_record(self, seq: int) -> Dict:
        """Return a dict containing metadata row + geoparsed content for given Seq value."""
        row_df = self.df[self.df["Seq"] == seq]
        if row_df.empty:
            raise ValueError(f"Seq {seq} not found in metadata.")
        row = row_df.iloc[0]
        record = row.to_dict()

        geo_fname = str(row.get("Geoparsed Filename", "")).strip()
        if not geo_fname or geo_fname.lower() == "nan":
            record["geoparsed"] = None
            return record

        # Ensure filename ends with .xml
        if not geo_fname.lower().endswith(".xml"):
            geo_fname = f"{geo_fname}.xml"

        xml_path = self.geoparsed_dir / geo_fname
        if not xml_path.exists():
            record["geoparsed"] = None
            record["geoparsed_error"] = f"File not found: {xml_path}"
            return record

        geoparsed_data = self._parse_geoparsed_xml(xml_path)
        record["geoparsed"] = geoparsed_data
        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_geoparsed_xml(xml_path: Path) -> Dict:
        """Extract full text and list of place entities (name, lat, lon) from a CQP-style geoparsed TEI file."""
        parser = ET.XMLParser(encoding="utf-8", recover=True)
        tree = ET.parse(str(xml_path), parser)
        root = tree.getroot()

        # Remove head elements before extracting text
        for head_elem in root.xpath("//head"):
            head_elem.getparent().remove(head_elem)

        # Extract full plain text (excluding head content)
        full_text = " ".join(root.itertext())
        full_text = " ".join(full_text.split())  # collapse whitespace

        # Extract entities
        entities: List[Dict[str, str]] = []
        for el in root.iter():
            if el.tag.lower().endswith("enamex"):
                name = (el.text or "").strip()
                lat = el.attrib.get("lat") or el.attrib.get("latitude")
                long = el.attrib.get("long") or el.attrib.get("longitude")
                entity_type = el.attrib.get("type")
                entities.append({
                    "name": name,
                    "lat": lat,
                    "long": long,
                    "type": entity_type,
                })

        return {"text": full_text, "entities": entities}


__all__ = ["DataLoader"]
