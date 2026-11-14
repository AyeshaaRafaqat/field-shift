"""Plain-language explanations for each rotation card."""
from __future__ import annotations

from .scoring import RotationScore


def why(o: RotationScore, current: RotationScore) -> list[str]:
    """Why this rotation was suggested, compared with what the farmer grows now."""
    if o.key == current.key:
        return ["This is the baseline: what you grow now. Alternatives are compared against it."]
    bits = []
    saving = current.irr - o.irr
    if saving > 50:
        bits.append(f"needs about {saving:.0f} mm/yr less irrigation than your current rotation")
    elif saving < -50:
        bits.append(f"needs about {-saving:.0f} mm/yr MORE irrigation than your current rotation")
    if o.fixer and not current.fixer:
        bits.append("adds a nitrogen-fixing legume")
    if o.pest > 0:
        bits.append("changes crop family, which helps break pest and disease cycles")
    if o.cover > current.cover + 0.03:
        bits.append(f"keeps the soil covered about {(o.cover - current.cover) * 365:.0f} more days a year")
    if o.resil > current.resil + 0.3:
        bits.append("faced less heat, frost and dry-spell stress in the 10-year record")
    if o.income > current.income + 0.3:
        bits.append("has a higher rough income index")
    return bits or ["performs similarly to your current rotation on the criteria used"]


def benefits(o: RotationScore) -> list[str]:
    out = []
    if o.fixer:
        out.append("Legume fixes atmospheric nitrogen, which can reduce fertiliser need")
    if (o.K.root + o.R.root) / 2 >= 3.5:
        out.append("Deep roots can improve infiltration and add organic matter below the surface")
    if (o.K.residue + o.R.residue) / 2 >= 3.5:
        out.append("Good residue cover protects soil from erosion and feeds soil organic matter")
    if o.pest > 0:
        out.append("Crop-family change interrupts pest and disease build-up")
    if o.irr < 500:
        out.append("Lower irrigation demand helps conserve groundwater")
    return out or ["No standout soil benefit from the criteria used"]


def risks(o: RotationScore, soil: str, limit: float) -> list[str]:
    """Risks and gaps that need local review."""
    out = [f"{o.K.name}: {o.K.notes}", f"{o.R.name}: {o.R.notes}"]
    if o.irr > limit:
        out.append(f"Estimated irrigation ({o.irr:.0f} mm/yr) exceeds your stated water limit ({limit} mm/yr)")
    if soil == "Clay loam" and ({o.K.name, o.R.name} & {"Chickpea", "Cotton"}):
        out.append("Heavy soil: check waterlogging and drainage")
    if soil == "Sandy loam":
        out.append("Sandy soil drains fast: irrigation need was raised 15%, and nutrients leach more easily")
    out.append("Confirm seed, machinery, labour and market access locally")
    return out
