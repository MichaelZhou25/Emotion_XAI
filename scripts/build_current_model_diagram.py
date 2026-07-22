import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


FONT = "fontFamily=Times New Roman;"
BOX = (
    "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
    "strokeWidth=1.5;fontSize=14;" + FONT
)
SMALL_BOX = BOX + "fontSize=13;"
TEXT = "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;" + FONT
EDGE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
    "jettySize=auto;html=1;endArrow=block;endFill=1;"
    "strokeWidth=1.5;strokeColor=#5B7083;fontSize=12;" + FONT
)
DASHED_EDGE = EDGE + "dashed=1;dashPattern=6 4;"


class DiagramBuilder:
    def __init__(self, diagram, page_width, page_height):
        diagram.clear()
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(page_width),
                "pageHeight": str(page_height),
                "math": "1",
                "shadow": "0",
                "background": "#FFFFFF",
            },
        )
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.counter = 0

    def _id(self, prefix):
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def box(self, value, x, y, width, height, fill, stroke="#5B7083", style="", cell_id=None):
        cell_id = cell_id or self._id("box")
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "parent": "1",
                "value": value,
                "vertex": "1",
                "style": BOX
                + f"fillColor={fill};strokeColor={stroke};"
                + style,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )
        return cell_id

    def text(self, value, x, y, width, height, style="", cell_id=None):
        cell_id = cell_id or self._id("text")
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "parent": "1",
                "value": value,
                "vertex": "1",
                "style": TEXT + style,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )
        return cell_id

    def band(self, title, x, y, width, height, fill, stroke, cell_id=None):
        cell_id = self.box(
            "",
            x,
            y,
            width,
            height,
            fill,
            stroke,
            "rounded=1;arcSize=5;dashed=1;dashPattern=8 5;connectable=0;",
            cell_id,
        )
        self.text(
            f"<b>{title}</b>",
            x + 16,
            y + 8,
            width - 32,
            30,
            "fontSize=18;fontStyle=1;align=left;verticalAlign=middle;",
        )
        return cell_id

    def edge(self, source, target, value="", style="", cell_id=None, points=None):
        cell_id = cell_id or self._id("edge")
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "parent": "1",
                "source": source,
                "target": target,
                "value": value,
                "edge": "1",
                "style": EDGE + style,
            },
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            point_array = ET.SubElement(geometry, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(point_array, "mxPoint", {"x": str(x), "y": str(y)})
        return cell_id


def title(builder, heading, subtitle, width):
    builder.text(
        heading,
        35,
        18,
        width - 70,
        42,
        "fontSize=26;fontStyle=1;align=center;verticalAlign=middle;fontColor=#16324F;",
    )
    builder.text(
        subtitle,
        35,
        60,
        width - 70,
        30,
        "fontSize=13;align=center;verticalAlign=middle;fontColor=#526777;",
    )


def build_page_1(diagram):
    b = DiagramBuilder(diagram, 1900, 1120)
    diagram.set("name", "Page-1 Overall Model")
    diagram.set("id", "eagle-current-overall")
    title(
        b,
        "<b>HemiMVEAGLE-Net: Overall Architecture</b>",
        "Current code: five-view feature fusion + Direct / hierarchy-aware HPCL / Edge Evidence + subject-adversarial learning",
        1900,
    )

    b.band("Input", 25, 120, 235, 690, "#F7F9FB", "#9AAAB7")
    input_box = b.box(
        "<b>LDS-DE features</b><br><br>"
        "<b>X</b> in R<sup>B x T x 62 x 5</sup><br><br>"
        "SEED: T = 30<br>SEED-IV: T = 10<br><br>"
        "5 bands: delta, theta,<br>alpha, beta, gamma",
        52,
        330,
        180,
        230,
        "#FFFFFF",
        "#4E7FA8",
        "fontSize=15;",
        "p1_input",
    )

    b.band("Multi-view Feature Extraction  (detailed in Page 2)", 280, 120, 710, 690, "#F5FAFD", "#4E7FA8")
    time = b.box(
        "<b>Time View</b><br>Linear + LN<br>TemporalEncoder<br><b>H<sub>time</sub></b> [B,T,d]",
        310,
        185,
        190,
        95,
        "#E8F1FA",
        "#4E7FA8",
        cell_id="p1_time",
    )
    freq = b.box(
        "<b>Frequency View</b><br>Linear + LN<br>TokenEncoder<br><b>H<sub>freq</sub></b> [B,5,d]",
        310,
        315,
        190,
        95,
        "#EAF6EE",
        "#4F8B66",
        cell_id="p1_freq",
    )
    channel = b.box(
        "<b>Channel View</b><br>Linear + LN<br>TokenEncoder<br><b>H<sub>channel</sub></b> [B,62,d]",
        310,
        445,
        190,
        95,
        "#FFF4E8",
        "#C98545",
        cell_id="p1_channel",
    )
    hemi = b.box(
        "<b>Hemispheric Fusion</b><br>27 L/R pairs<br>D, |D|, S<br><b>H<sub>hemi</sub></b> [B,27,d]",
        310,
        575,
        190,
        105,
        "#F3ECFA",
        "#8660A8",
        cell_id="p1_hemi",
    )
    region = b.box(
        "<b>Region Aggregation</b><br>5 predefined regions<br>pair-token mean<br><b>H<sub>region</sub></b> [B,5,d]",
        310,
        710,
        190,
        85,
        "#FCEEF2",
        "#B45E78",
        cell_id="p1_region",
    )

    pooled = []
    for key, y, fill, stroke, source in [
        ("time", 190, "#E8F1FA", "#4E7FA8", time),
        ("freq", 320, "#EAF6EE", "#4F8B66", freq),
        ("channel", 450, "#FFF4E8", "#C98545", channel),
        ("hemi", 585, "#F3ECFA", "#8660A8", hemi),
        ("region", 710, "#FCEEF2", "#B45E78", region),
    ]:
        pool = b.box(
            "<b>AttnPool</b>",
            545,
            y + 12,
            115,
            55,
            "#FFFFFF",
            stroke,
            cell_id=f"p1_{key}_pool",
        )
        z = b.box(
            f"<b>z<sub>{key}</sub></b><br>[B,d]",
            700,
            y + 7,
            105,
            65,
            fill,
            stroke,
            cell_id=f"p1_z_{key}",
        )
        b.edge(source, pool)
        b.edge(pool, z)
        pooled.append(z)

    gate = b.box(
        "<b>Gated View Fusion</b><br>"
        "Stack [B,5,d]<br>Linear(d,1) + Softmax<br>"
        "weighted sum",
        820,
        380,
        150,
        135,
        "#FFF8DB",
        "#B29A32",
        "fontSize=13;",
        "p1_gate",
    )
    fused = b.box(
        "<b>z<sub>fused</sub></b><br>[B,d]",
        835,
        585,
        120,
        70,
        "#FFF1C7",
        "#B29A32",
        "fontSize=16;",
        "p1_fused",
    )
    for index, z in enumerate(pooled):
        b.edge(z, gate, style=f"entryX=0;entryY={(index + 1) / 6:.3f};")
    b.edge(gate, fused)
    b.edge(channel, hemi, "L/R pairs", style="dashed=1;dashPattern=6 4;")
    b.edge(hemi, region, "pair groups", style="dashed=1;dashPattern=6 4;")
    for target in (time, freq, channel):
        b.edge(input_box, target)

    b.band("Prediction and Representation Branches", 1020, 120, 520, 690, "#FCFCFD", "#7C8792")
    direct = b.box(
        "<b>Direct Classification</b><br>"
        "LayerNorm + Dropout + Linear<br>"
        "<b>logits<sub>direct</sub></b> [B,K]",
        1050,
        175,
        270,
        105,
        "#E8F1FA",
        "#4E7FA8",
        cell_id="p1_direct",
    )
    direct_logits = b.box(
        "<b>Direct</b><br>[B,K]",
        1360,
        190,
        145,
        70,
        "#FFFFFF",
        "#4E7FA8",
        cell_id="p1_direct_logits",
    )
    hpcl = b.box(
        "<b>Hierarchy-aware HPCL</b><br>"
        "Project to d<sub>proto</sub> -> ExpMap<sub>0</sub><br>"
        "Poincare distances to hierarchy prototypes<br>"
        "Prototype CE + HNCE + hierarchy margin + radius",
        1050,
        330,
        270,
        150,
        "#F3ECFA",
        "#8660A8",
        "fontSize=13;",
        "p1_hpcl",
    )
    proto_logits = b.box(
        "<b>Prototype</b><br>logits [B,K]<br>z<sub>hyp</sub>, distances",
        1360,
        370,
        145,
        85,
        "#FFFFFF",
        "#8660A8",
        cell_id="p1_proto_logits",
    )
    edge_branch = b.box(
        "<b>Multi-view Edge Evidence</b><br>"
        "Tokens: H<sub>time</sub> + H<sub>freq</sub> + H<sub>hemi</sub> + H<sub>region</sub><br>"
        "[B, T+37, d]<br>"
        "E edge queries -> attention -> evidence -> Sigmoid",
        1050,
        530,
        270,
        150,
        "#EAF6EE",
        "#4F8B66",
        "fontSize=13;",
        "p1_edge",
    )
    edge_logits = b.box(
        "<b>Edge</b><br>logits [B,K]<br>edge weights + XAI maps",
        1360,
        565,
        145,
        85,
        "#FFFFFF",
        "#4F8B66",
        cell_id="p1_edge_logits",
    )
    domain = b.box(
        "<b>Subject-Invariant Learning</b><br>"
        "GRL (lambda = 0.2)<br>Subject Discriminator<br>"
        "<b>domain logits</b> [B,15]",
        1050,
        710,
        270,
        90,
        "#FCEBEC",
        "#B85C5C",
        cell_id="p1_domain",
    )
    domain_logits = b.box(
        "<b>L<sub>domain</sub></b><br>subject CE",
        1360,
        720,
        145,
        70,
        "#FFFFFF",
        "#B85C5C",
        cell_id="p1_domain_logits",
    )
    b.edge(fused, direct, points=[(1005, 620), (1005, 228)])
    b.edge(fused, hpcl, points=[(1005, 620), (1005, 405)])
    b.edge(fused, domain, points=[(1005, 620), (1005, 755)])
    b.edge(direct, direct_logits)
    b.edge(hpcl, proto_logits)
    b.edge(edge_branch, edge_logits)
    b.edge(domain, domain_logits)
    edge_tokens = b.box(
        "<b>Concat edge tokens</b><br>"
        "H<sub>time</sub>, H<sub>freq</sub>, H<sub>hemi</sub>, H<sub>region</sub><br>"
        "[B,T+37,d]",
        815,
        705,
        155,
        80,
        "#EAF6EE",
        "#4F8B66",
        "fontSize=12;",
        "p1_edge_tokens",
    )
    b.edge(
        edge_tokens,
        edge_branch,
        "token bypass",
        style="dashed=1;dashPattern=6 4;",
        points=[(1005, 745), (1005, 605)],
    )

    fusion = b.box(
        "<b>Logit Fusion</b><br><br>"
        "1.0 Direct<br>+ 0.3 Prototype<br>+ 0.2 Edge",
        1580,
        350,
        150,
        160,
        "#FFF8DB",
        "#B29A32",
        "fontSize=15;",
        "p1_logit_fusion",
    )
    output = b.box(
        "<b>Final Prediction</b><br><br>"
        "<b>logits</b> [B,K]<br><br>"
        "SEED: K=3<br>SEED-IV: K=4",
        1760,
        350,
        120,
        160,
        "#E8F1FA",
        "#2D678F",
        "fontSize=14;",
        "p1_output",
    )
    for source in (direct_logits, proto_logits, edge_logits):
        b.edge(source, fusion)
    b.edge(fusion, output)

    b.band("Training Objective  (Concept branch disabled)", 25, 850, 1855, 235, "#FAFAFB", "#8A98A5")
    loss = b.box(
        "<b>Total loss</b><br><br>"
        "L = L<sub>final CE</sub> + 1.0 L<sub>direct CE</sub> + 0.2 L<sub>proto CE</sub> "
        "+ 0.1 L<sub>edge BCE</sub> + 0.02 L<sub>consistency</sub> + 0.001 L<sub>radius</sub><br>"
        "+ 0.02 L<sub>hierarchy prototype NCE</sub> + 0.02 L<sub>hierarchy margin</sub> "
        "+ 0.05 L<sub>domain</sub>",
        55,
        915,
        1160,
        115,
        "#FFFFFF",
        "#647789",
        "fontSize=15;align=left;spacingLeft=14;",
        "p1_loss",
    )
    notes = b.box(
        "<b>Interpretability outputs</b><br><br>"
        "view weights | time/frequency/channel attention<br>"
        "pair and region attention | edge-to-token attention<br>"
        "Poincare prototype distances",
        1250,
        915,
        590,
        115,
        "#FFFFFF",
        "#4F8B66",
        "fontSize=14;align=left;spacingLeft=14;",
        "p1_xai",
    )


def build_page_2(diagram):
    b = DiagramBuilder(diagram, 2000, 1220)
    diagram.set("name", "Page-2 Feature Extraction")
    diagram.set("id", "eagle-current-feature-extraction")
    title(
        b,
        "<b>Multi-view Feature Extraction and Gated Fusion</b>",
        "Exact tensor transformations in HemiMVEAGLENet.forward | C=62, F=5, d=128, L=27 bilateral pairs, R=5 regions",
        2000,
    )

    input_box = b.box(
        "<b>Input X</b><br><br>[B,T,62,5]<br><br>"
        "T=30 (SEED)<br>T=10 (SEED-IV)",
        35,
        430,
        185,
        150,
        "#FFFFFF",
        "#2D678F",
        "fontSize=16;",
        "p2_input",
    )

    b.band("Time View", 260, 105, 1280, 175, "#F5FAFD", "#4E7FA8")
    time_reshape = b.box(
        "<b>Reshape</b><br>[B,T,62x5]<br>[B,T,310]",
        290,
        160,
        160,
        80,
        "#FFFFFF",
        "#4E7FA8",
        cell_id="p2_time_reshape",
    )
    time_embed = b.box(
        "<b>Time embedding</b><br>Linear(310,d)<br>LayerNorm",
        500,
        160,
        170,
        80,
        "#E8F1FA",
        "#4E7FA8",
        cell_id="p2_time_embed",
    )
    time_encoder = b.box(
        "<b>TemporalEncoder</b><br>temporal context<br>modeling",
        720,
        160,
        180,
        80,
        "#E8F1FA",
        "#4E7FA8",
        cell_id="p2_time_encoder",
    )
    h_time = b.box(
        "<b>H<sub>time</sub></b><br>[B,T,d]",
        950,
        160,
        135,
        80,
        "#FFFFFF",
        "#4E7FA8",
        cell_id="p2_h_time",
    )
    time_pool = b.box(
        "<b>AttnPool</b><br>time_attention [B,T]",
        1135,
        160,
        185,
        80,
        "#E8F1FA",
        "#4E7FA8",
        cell_id="p2_time_pool",
    )
    z_time = b.box(
        "<b>z<sub>time</sub></b><br>[B,d]",
        1370,
        160,
        135,
        80,
        "#FFFFFF",
        "#4E7FA8",
        cell_id="p2_z_time",
    )

    b.band("Frequency View", 260, 305, 1280, 175, "#F7FBF8", "#4F8B66")
    freq_reshape = b.box(
        "<b>Permute + flatten</b><br>[B,5,T,62]<br>[B,5,62T]",
        290,
        360,
        160,
        80,
        "#FFFFFF",
        "#4F8B66",
        cell_id="p2_freq_reshape",
    )
    freq_embed = b.box(
        "<b>Frequency embedding</b><br>Linear(62T,d)<br>LayerNorm",
        500,
        360,
        170,
        80,
        "#EAF6EE",
        "#4F8B66",
        cell_id="p2_freq_embed",
    )
    freq_encoder = b.box(
        "<b>TokenEncoder</b><br>1-layer Transformer<br>over 5 bands",
        720,
        360,
        180,
        80,
        "#EAF6EE",
        "#4F8B66",
        cell_id="p2_freq_encoder",
    )
    h_freq = b.box(
        "<b>H<sub>freq</sub></b><br>[B,5,d]",
        950,
        360,
        135,
        80,
        "#FFFFFF",
        "#4F8B66",
        cell_id="p2_h_freq",
    )
    freq_pool = b.box(
        "<b>AttnPool</b><br>freq_attention [B,5]",
        1135,
        360,
        185,
        80,
        "#EAF6EE",
        "#4F8B66",
        cell_id="p2_freq_pool",
    )
    z_freq = b.box(
        "<b>z<sub>freq</sub></b><br>[B,d]",
        1370,
        360,
        135,
        80,
        "#FFFFFF",
        "#4F8B66",
        cell_id="p2_z_freq",
    )

    b.band("Channel View", 260, 505, 1280, 175, "#FFF9F2", "#C98545")
    channel_reshape = b.box(
        "<b>Permute + flatten</b><br>[B,62,T,5]<br>[B,62,5T]",
        290,
        560,
        160,
        80,
        "#FFFFFF",
        "#C98545",
        cell_id="p2_channel_reshape",
    )
    channel_embed = b.box(
        "<b>Channel embedding</b><br>Linear(5T,d)<br>LayerNorm",
        500,
        560,
        170,
        80,
        "#FFF4E8",
        "#C98545",
        cell_id="p2_channel_embed",
    )
    channel_encoder = b.box(
        "<b>TokenEncoder</b><br>1-layer Transformer<br>over 62 channels",
        720,
        560,
        180,
        80,
        "#FFF4E8",
        "#C98545",
        cell_id="p2_channel_encoder",
    )
    h_channel = b.box(
        "<b>H<sub>channel</sub></b><br>[B,62,d]",
        950,
        560,
        135,
        80,
        "#FFFFFF",
        "#C98545",
        cell_id="p2_h_channel",
    )
    channel_pool = b.box(
        "<b>AttnPool</b><br>channel_attention [B,62]",
        1135,
        560,
        185,
        80,
        "#FFF4E8",
        "#C98545",
        cell_id="p2_channel_pool",
    )
    z_channel = b.box(
        "<b>z<sub>channel</sub></b><br>[B,d]",
        1370,
        560,
        135,
        80,
        "#FFFFFF",
        "#C98545",
        cell_id="p2_z_channel",
    )

    b.band("Hemispheric Evidence Fusion", 260, 705, 1280, 185, "#FAF7FD", "#8660A8")
    pair_select = b.box(
        "<b>27 predefined pairs</b><br>H<sub>L</sub>, H<sub>R</sub><br>[B,27,d]",
        290,
        765,
        160,
        85,
        "#FFFFFF",
        "#8660A8",
        cell_id="p2_pair_select",
    )
    das = b.box(
        "<b>Bilateral evidence</b><br>"
        "D = H<sub>L</sub> - H<sub>R</sub><br>"
        "A = |H<sub>L</sub> - H<sub>R</sub>|<br>"
        "S = H<sub>L</sub> + H<sub>R</sub>",
        500,
        755,
        210,
        105,
        "#F3ECFA",
        "#8660A8",
        "fontSize=13;",
        "p2_das",
    )
    hemi_project = b.box(
        "<b>Concat(D,A,S)</b><br>[B,27,3d]<br>Linear(3d,d)<br>LN + GELU",
        760,
        755,
        180,
        105,
        "#F3ECFA",
        "#8660A8",
        cell_id="p2_hemi_project",
    )
    h_hemi = b.box(
        "<b>H<sub>hemi</sub></b><br>[B,27,d]",
        990,
        765,
        135,
        85,
        "#FFFFFF",
        "#8660A8",
        cell_id="p2_h_hemi",
    )
    pair_pool = b.box(
        "<b>Pair AttnPool</b><br>pair_attention [B,27]",
        1175,
        765,
        170,
        85,
        "#F3ECFA",
        "#8660A8",
        cell_id="p2_pair_pool",
    )
    z_hemi = b.box(
        "<b>z<sub>hemi</sub></b><br>[B,d]",
        1390,
        765,
        115,
        85,
        "#FFFFFF",
        "#8660A8",
        cell_id="p2_z_hemi",
    )

    b.band("Region Evidence Aggregation", 260, 915, 1280, 185, "#FDF7F9", "#B45E78")
    groups = b.box(
        "<b>5 region groups</b><br>frontal / temporal /<br>central / parietal / occipital",
        390,
        970,
        200,
        90,
        "#FFFFFF",
        "#B45E78",
        "fontSize=13;",
        "p2_groups",
    )
    mean = b.box(
        "<b>Mean over pair tokens</b><br>within each region",
        650,
        970,
        180,
        90,
        "#FCEEF2",
        "#B45E78",
        cell_id="p2_region_mean",
    )
    h_region = b.box(
        "<b>H<sub>region</sub></b><br>[B,5,d]",
        890,
        970,
        145,
        90,
        "#FFFFFF",
        "#B45E78",
        cell_id="p2_h_region",
    )
    region_pool = b.box(
        "<b>Region AttnPool</b><br>region_attention [B,5]",
        1095,
        970,
        190,
        90,
        "#FCEEF2",
        "#B45E78",
        cell_id="p2_region_pool",
    )
    z_region = b.box(
        "<b>z<sub>region</sub></b><br>[B,d]",
        1345,
        970,
        140,
        90,
        "#FFFFFF",
        "#B45E78",
        cell_id="p2_z_region",
    )

    b.band("Gated Fusion", 1580, 105, 385, 995, "#FFFCF0", "#B29A32")
    stack = b.box(
        "<b>Stack five view vectors</b><br><br>"
        "Z<sub>views</sub> = [z<sub>time</sub>, z<sub>freq</sub>, "
        "z<sub>channel</sub>, z<sub>hemi</sub>, z<sub>region</sub>]<br><br>"
        "[B,5,d]",
        1630,
        300,
        285,
        175,
        "#FFFFFF",
        "#B29A32",
        "fontSize=15;",
        "p2_stack",
    )
    view_logits = b.box(
        "<b>View gate</b><br>Linear(d,1) applied per view<br>"
        "view_logits [B,5]",
        1630,
        530,
        285,
        105,
        "#FFF8DB",
        "#B29A32",
        cell_id="p2_view_logits",
    )
    view_weights = b.box(
        "<b>Softmax over views</b><br>view_weights [B,5]<br>"
        "interpretable contribution scores",
        1630,
        690,
        285,
        115,
        "#FFF8DB",
        "#B29A32",
        cell_id="p2_view_weights",
    )
    z_fused = b.box(
        "<b>Weighted sum</b><br><br>"
        "z<sub>fused</sub> = Sum(v) w<sub>v</sub> z<sub>v</sub><br><br>"
        "<b>[B,d]</b>",
        1630,
        860,
        285,
        145,
        "#FFF1C7",
        "#B29A32",
        "fontSize=16;",
        "p2_z_fused",
    )
    b.text(
        "Current setting: <b>use_view_identity_embeddings = false</b>",
        1595,
        1040,
        355,
        35,
        "fontSize=13;align=center;fontColor=#6D6229;",
    )

    for target in (time_reshape, freq_reshape, channel_reshape):
        b.edge(input_box, target)
    for row in [
        (time_reshape, time_embed, time_encoder, h_time, time_pool, z_time),
        (freq_reshape, freq_embed, freq_encoder, h_freq, freq_pool, z_freq),
        (channel_reshape, channel_embed, channel_encoder, h_channel, channel_pool, z_channel),
        (pair_select, das, hemi_project, h_hemi, pair_pool, z_hemi),
        (groups, mean, h_region, region_pool, z_region),
    ]:
        for source, target in zip(row, row[1:]):
            b.edge(source, target)
    b.edge(
        h_channel,
        pair_select,
        "bilateral indices",
        style="dashed=1;dashPattern=6 4;",
        points=[(1018, 690), (335, 690)],
    )
    b.edge(
        h_hemi,
        groups,
        "region indices",
        style="dashed=1;dashPattern=6 4;",
        points=[(1058, 900), (490, 900)],
    )
    fusion_routes = [
        (z_time, 0.15, 200),
        (z_freq, 0.32, 400),
        (z_channel, 0.50, 600),
        (z_hemi, 0.68, 808),
        (z_region, 0.85, 1015),
    ]
    for source, entry_y, source_y in fusion_routes:
        target_y = 300 + 175 * entry_y
        b.edge(
            source,
            stack,
            style=f"entryX=0;entryY={entry_y};",
            points=[(1555, source_y), (1555, target_y)],
        )
    b.edge(stack, view_logits)
    b.edge(view_logits, view_weights)
    b.edge(view_weights, z_fused)


def main():
    parser = argparse.ArgumentParser(description="Build the current two-page EAGLE-Net draw.io diagram.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tree = ET.parse(args.source)
    root = tree.getroot()
    diagrams = root.findall("diagram")
    if len(diagrams) != 2:
        raise ValueError(f"Expected exactly two pages, found {len(diagrams)}")

    root.set("host", "app.diagrams.net")
    root.set("pages", "2")
    root.set("modified", "2026-07-18T00:00:00.000Z")
    root.set("agent", "EAGLE-Net diagram updater")
    root.set("version", "current")

    build_page_1(diagrams[0])
    build_page_2(diagrams[1])
    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output)


if __name__ == "__main__":
    main()
