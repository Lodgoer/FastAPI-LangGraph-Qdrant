# Document Parsing and Vector Search for a RAG Project

*Notes from a long learning session around the `FastAPI-LangGraph-Qdrant` project: what the project does, how to parse hard PDFs (Docling vs Landing AI vs LlamaParse), how Qdrant stores data, and how HNSW search works.*

Last updated: 17 October 2026

---

## A quick word before you read

This started as "help me understand the project's flow document" and turned into something broader: *how do you get messy real-world PDFs into a vector database, and how do you search them well?*

The project flow part is kept short on purpose. The parts that took most of the thinking are the parsers comparison, the Qdrant schema and HNSW, and the sources list at the bottom.

A few ground rules I tried to follow:

- Anything I read on an official page during the session is linked in the sources.
- Anything I'm recalling from memory is flagged as unverified.
- Anything that's my own engineering opinion is called out as such.

---

## 1. The project in brief

The document being studied: <https://github.com/AI-Projects-list/FastAPI-LangGraph-Qdrant/blob/main/PROJECT_FLOW.md>

It's a small RAG service: you can **store documents** (text or PDF) as searchable vectors and then **ask questions** about them. It's built with FastAPI, LangGraph and Qdrant in a clean/hexagonal architecture.

### The layers, in one picture

```
Client → FastAPI → Application Services → Domain (Ports) → Infrastructure (Adapters)
```

- **FastAPI** is the front door (routes, validation).
- **Application services** (`DocumentService`, `QueryService`) decide what happens and in what order.
- **Ports** are interfaces (like `IVectorStore` in C#). They describe what's needed, not how.
- **Adapters** are the real implementations: the Qdrant store, the LangGraph workflow, the embeddings service, an in-memory state store.

The key idea is **dependency inversion**: the code depends on the interfaces, not on Qdrant or LangGraph directly. That makes it easy to swap a part and easy to test with fakes.

### The four flows

| Flow | What happens |
|---|---|
| **Startup** | FastAPI's lifespan builds a container once: loads settings from `ENVIRONMENT` (dev/staging/prod), connects to Qdrant, creates the collection if missing, wires up the services, registers routers. |
| **Ingest text** | `POST /ingest` → embed the text → make a UUID and payload → save to the state store → upsert the point into Qdrant → return the id. |
| **Upload PDF** | Check it's a PDF → read pages with PyPDF2 → for each page with text: embed, UUID, store; pages without text are `skipped` → return a per-page summary. |
| **Query (RAG)** | Embed the question → retrieve nearest points from Qdrant → add them to the prompt → LLM answers. (Built as a LangGraph graph; not walked through in detail yet.) |

### Terms worth knowing

| Term | One-line meaning |
|---|---|
| Embedding | Text turned into a list of numbers, so similar meanings sit close together |
| Vector database | A database for "find the most similar items" instead of exact matches |
| Collection / Point / Payload | Qdrant's table / row / free-form JSON attached to each row |
| Upsert | Insert, or replace if the id already exists |
| Lifespan / Container / DI | Startup hook / one object holding all services / handing dependencies in from outside |
| Chunk | A piece of text embedded on its own (here: one PDF page) |

### What I'd flag in that document

1. **Embeddings are random** (`RandomEmbeddingsService`), so search results aren't semantically meaningful yet.
2. **The route does too much** during ingest, and the state store is saved twice.
3. **Saves aren't atomic.** If Qdrant fails after the state store succeeds, the two disagree.
4. **PDF handling is naive.** One page equals one chunk, the file-type check trusts the client, and scanned pages are silently skipped. The next section is about this.
5. **Config oddities.** The Qdrant port mapping (`6334:6333`) is confusing, and "single instance per request" contradicts the startup singleton.

That's all I'll say about the project document. Everything below is the more reusable material.

---

## 2. Scanned pages, charts and tables: what to do instead of `extract_text()`

Plain text extraction fails in three ways:

| Page type | What goes wrong |
|---|---|
| Scanned (page is one big image) | No text found, page skipped |
| Mixed (text plus charts or photos) | The chart's content is lost |
| Tables | Text comes out jumbled; column order is gone |

There's no single magic tool, but there *is* a sensible shape for a solution:

```
PDF → ① Triage → ② Layout-aware parsing (+OCR) → ③ Enrichment → ④ Chunking → Embed → Qdrant
```

### ① Triage (my own suggestion, not from a paper)

Decide per page whether it's scanned, mixed or plain text, so the cheap path stays cheap. The thresholds are guesses you'll need to tune on your own PDFs:

```python
import fitz  # PyMuPDF

def classify_page(page) -> str:
    text_chars = len(page.get_text().strip())
    page_area = page.rect.width * page.rect.height
    img_area = sum(fitz.Rect(i["bbox"]).get_area() for i in page.get_image_info())
    img_ratio = img_area / page_area

    if text_chars < 30 and img_ratio > 0.5:
        return "scanned"
    if img_ratio > 0.15:
        return "mixed"
    return "native_text"
```

### ② Layout-aware parsing

A vision model looks at the page and labels regions: heading, paragraph, table, figure, caption. You get structure instead of one tangled string. Section 3 compares three tools that do this.

### ③ Enrichment: turning visuals into searchable text

A text embedding model can't read the pixels of a chart, so convert visual content into text:

- **Figures and charts:** crop the figure, ask a vision-language model to describe it, and ideally to output the underlying numbers as a small table. Embed the description plus the caption, and keep the image path in the payload.
- **Tables:** convert to Markdown or HTML, then also write a short natural-language summary and embed *that*. Keep the full table in the payload for the LLM.

### ④ Structure-aware chunking

Chunk by section headings, never split a table or figure in half, add a little overlap, and prepend the section title to each chunk.

### A different route: embed the page image itself

**ColPali** skips OCR and layout parsing and embeds the page image directly with a vision-language model, matching queries against it using late interaction (many vectors per page). It also introduced the ViDoRe benchmark.

- Upside: charts and layout are captured naturally.
- Downsides (my understanding, not verified): each page becomes hundreds of vectors, so storage is heavy; answering needs a multimodal LLM; and language coverage matters, since the original work focused on English and French.

---

## 3. Docling vs Landing AI vs LlamaParse

This section exists because these three came up in a working session. **I wasn't in that session**, so I don't know what was said there. What follows is based on the official pages for each tool (linked in the sources), plus what I read earlier about Docling. If your session notes disagree with anything here, trust your notes and the live docs over me.

### The one-line version of each

- **Docling** is an open-source library (MIT license) from IBM that you run yourself. It combines a layout model, a table-structure model (TableFormer) and OCR, and you choose the OCR engine and enrichment options.
- **Landing AI Parse API** is a hosted service. Its model is a *Document Pre-Trained Transformer* (DPT). According to the docs, it identifies the layout, writes the content out as Markdown in reading order, and records where every block sits on the page.
- **LlamaParse (Parse)** is the hosted document-parsing product from LlamaIndex. It turns PDFs, scans, tables and charts into Markdown, text or JSON. Its docs say it uses generative models for end-to-end understanding rather than a bounding-box detector bolted onto OCR. That's the vendor's own description, so treat it as a claim to test.

### Side by side

| | **Docling** | **Landing AI (DPT-3)** | **LlamaParse** |
|---|---|---|---|
| **How you use it** | Python library you run locally or on your own server | Hosted API (Parse and async Parse Jobs), plus Python/TypeScript libraries and a Playground | Hosted API with SDKs (Python, TypeScript, Go, Java), CLI and web UI; the docs also list self-hosting / BYOC |
| **Core approach** | Specialized models (layout, tables) plus an OCR engine you pick | One document transformer reads layout and transcribes to Markdown | Generative, "end-to-end" parsing (vendor description) |
| **Your main dials** | OCR engine, enrichment options, which VLM describes pictures | Model choice: **Pro** (best quality, default) or **Verity** (faster, cheaper, preview) | Tier: **Agentic Plus**, **Agentic**, **Cost Effective**, **Fast** |
| **Scans and handwriting** | Via the OCR option (slower) | Pro yes; Verity no | OCR runs automatically on pages with no text layer; the `fast` tier runs no AI model |
| **Non-Latin scripts** | Depends on the OCR engine you choose | Pro yes; Verity no | OCR language list with 80+ codes; Arabic is listed, but I didn't see Persian named on the page I read |
| **Figures and charts** | Picture classification and description as optional enrichment | Pro: detailed figure descriptions; Verity: basic | Has dedicated charts-and-figures and tables pages (I only read the overview and OCR pages) |
| **Locating things on the page** | Layout elements with positions | Grounding down to the line (Pro) or word (Verity, with confidence scores) | Layout and bounding boxes feature, with word/line/cell grounding examples; per-page confidence for scans |
| **Cost control** | Your own compute | Credits; Verity uses fewer credits than Pro | Cost Optimizer routes simple pages to a cheaper tier and complex pages to the premium tier |
| **Reproducibility** | Pin the library version | Pin dated snapshots, e.g. `dpt-3-pro-20260710` | Versioned tiers you can pin |
| **Where your documents go** | Stay on your machine | Sent to the vendor's API | Sent to the vendor's API, unless you self-host |

### What actually differs, in plain words

**1. Local control versus a managed service.**
Docling is something you operate. You pay in your own time and hardware, but nothing leaves your machine, and you can inspect and swap each part. The other two are services: you send a file and get a result back, which means less engineering but also less control and a per-use bill.

**2. How much "model" is inside.**
Docling is a pipeline of specialized models that you can see and tune. Landing AI and LlamaParse describe more end-to-end, model-driven parsing. That tends to handle messy layouts gracefully, but it's harder to debug when something goes wrong.

**3. How each exposes the quality-versus-cost trade-off.**
Landing AI asks you to pick a *model* (Pro for hard documents, Verity for clean digital ones). LlamaParse asks you to pick a *tier*, and can mix tiers page by page automatically with Cost Optimizer. Docling's equivalent is deciding whether to turn OCR and picture description on.

**4. Which kinds of pages each is built for.**
Landing AI states the split explicitly: Verity is for digitally created text documents; Pro is for scans, handwriting, non-Latin scripts and complex forms. LlamaParse's recommendation is similar in spirit: the `agentic` tiers for scans and visually rich files, and `fast` for plain text at volume.

**5. Reproducibility.**
Both hosted services let you pin a specific version so results don't shift under you. With Docling you get the same effect by pinning the package version.

### Things I could not verify

- **Pricing** for either hosted service. Check their credit and pricing pages before estimating cost.
- **Persian quality** for all three. LlamaParse's page lists many languages but I didn't see Persian named; Landing AI says Pro handles non-Latin scripts but doesn't name languages on the page I read. Docling depends on whichever OCR engine you plug in. **Test with real Persian pages.**
- **Availability and billing in your region.** Hosted services based abroad sometimes limit certain countries or payment methods. I haven't checked this for either one, so check before you build around it. Docling doesn't have this problem, since it runs locally.
- **Head-to-head accuracy.** I don't have trustworthy benchmark numbers comparing them on your kind of documents. The vendors' own pages are marketing, not independent tests.

### How I'd choose (my opinion)

1. **Start with Docling.** It's free, local and private, and it already covers layout, tables, OCR and optional picture descriptions.
2. **Build a tiny benchmark.** Take about 20 pages from your real data (a scan, a table-heavy page, a chart page, a Persian page, a clean text page). Run all three tools on them and compare by eye.
3. **Use a hosted parser selectively.** If Docling struggles on a class of pages, send *only those* to LlamaParse or Landing AI rather than paying for everything.
4. **Hide the choice behind a port.** This is exactly what the project's architecture is for:

```python
# domain/ports.py
class DocumentParserPort(Protocol):
    def parse(self, pdf_bytes: bytes) -> list[ParsedElement]: ...
```

```
infrastructure/
  ├── docling_parser.py       ← DocumentParserPort (local)
  ├── llamaparse_parser.py    ← DocumentParserPort (hosted)
  ├── landingai_parser.py     ← DocumentParserPort (hosted)
  └── pypdf_parser.py         ← the old simple version, kept for tests
```

`DocumentService` only knows the port, so you can swap or combine parsers without touching it, and tests use a fake parser.

### Session notes (add your own here)

*What was said in your session about the differences, any numbers or demos, and what the team decided:*

-
-
-

---

## 4. How data is shaped in Qdrant

Qdrant isn't SQL: there's no fixed table schema. Only the *vector's shape* is fixed up front; the rest is free-form.

| SQL | Qdrant | Notes |
|---|---|---|
| Table | **Collection** | named group of points with the same vector settings |
| Row | **Point** | `{ id, vector, payload }` |
| Primary key | **ID** | unsigned 64-bit integer or UUID |
| numeric column | **Vector** | dense, sparse or multivector |
| other columns | **Payload** | free JSON, no enforced schema |
| Index | **Payload index / HNSW** | |

### The pieces

- **Collection:** you fix the vector **size** (must match your embedding model) and the **distance** (Cosine, Dot or Euclid). From my understanding, you can't change the size later without recreating the collection, so settle your embedding model first.
- **Vector kinds:** *dense* (a normal embedding), *sparse* (mostly zeros, good for keyword-style search), *multivector* (several vectors per point, for ColBERT and ColPali). You can also use *named vectors* to keep, say, a text vector and an image vector in one point.
- **Payload:** structured metadata with no enforced schema. That's flexible but risky: if one point says `page_number` and another says `page`, nothing complains and your filters quietly break. Keep the schema in your own code, for example a Pydantic model.
- **Payload index:** makes filtering on a field efficient. Index every field you filter on. Qdrant's docs note it's best to create these before loading data, because they extend the HNSW graph.

### Problems with the project's current schema

1. No parent `doc_id`, so you can't easily delete "all pages of this PDF".
2. No payload indexes, so filtering will be slow.
3. Nested `metadata`, which is clumsier to filter on than flat fields.
4. No `element_type` to distinguish text, tables and figures.

### A better shape

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="documents",
    vectors_config=models.VectorParams(
        size=384,   # must equal your embedding model's output size
        distance=models.Distance.COSINE,
    ),
)

for field, schema in [
    ("doc_id",       models.PayloadSchemaType.KEYWORD),
    ("filename",     models.PayloadSchemaType.KEYWORD),
    ("element_type", models.PayloadSchemaType.KEYWORD),
    ("page_number",  models.PayloadSchemaType.INTEGER),
    ("created_at",   models.PayloadSchemaType.DATETIME),
]:
    client.create_payload_index("documents", field_name=field, field_schema=schema)
```

```json
{
  "doc_id": "parent-file-uuid",
  "chunk_id": "this-chunk-uuid",
  "filename": "report.pdf",
  "page_number": 3,
  "element_type": "text | table | figure",
  "section": "3.2 Results",
  "content": "the chunk's text",
  "image_path": null,
  "created_at": "2026-10-16T10:00:00Z"
}
```

With `doc_id` in the payload, deleting a whole PDF is one filtered delete:

```python
client.delete(
    "documents",
    points_selector=models.FilterSelector(
        filter=models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        ])
    ),
)
```

(Method and class names come from my knowledge of `qdrant-client` and can differ between versions. Check them against what you've installed.)

**Memory note:** by default Qdrant keeps vectors in RAM for speed, which gets expensive at millions of vectors. Disk storage and quantization are the usual answers. Multivector setups like ColPali are much heavier, since each page becomes hundreds of vectors.

---

## 5. ANN, HNSW and retrieving results

### The problem

Given a query vector, find the k closest stored vectors (k-nearest neighbors). **Brute force** compares against everything: exact, but the cost grows linearly. By my rough arithmetic, a million 384-dimensional vectors means about 384 million multiplications per query.

### Approximate nearest neighbors (ANN)

Allow a tiny amount of error in exchange for a large speed-up. **Recall** measures the quality: of the true nearest neighbors, what fraction did you find? Finding 9 of 10 is a recall of 0.9. For RAG, around 0.95 or higher is typically comfortable.

### HNSW (Hierarchical Navigable Small World)

HNSW is the index Qdrant uses for dense vectors.

*An analogy (mine, not the paper's):* driving somewhere, you take the highway first, then main roads, then local streets.

- **Structure:** a layered graph. Every vector is in layer 0. A random subset also appears in layer 1, a smaller subset in layer 2, and so on. The top layer is sparse with long jumps; the bottom layer is dense with short links. Per the paper, each element's top layer is drawn randomly with an exponentially decaying probability.
- **Search:** start at an entry point in the top layer, hop greedily to whichever neighbor is closer to the query, drop a layer, repeat; on layer 0 keep a candidate list and return the best k.
- **Why it's fast:** starting from the top and separating distance scales gives roughly logarithmic scaling.
- **Cost:** building the graph takes time and memory, because each insert runs a search to find neighbors to link to.

### The three knobs

| Parameter | Set when | Meaning |
|---|---|---|
| `m` | creating the collection | links per node; higher tends to improve recall but grows the index and build time |
| `ef_construct` | creating the collection | candidate list size while building; larger means a better index but a slower build |
| `hnsw_ef` | each search | candidate list size at query time; higher is more accurate but slower |

According to a community write-up citing Qdrant's docs, `hnsw_ef` defaults to the collection's `ef_construct`, which defaults to 100, and if `limit` exceeds `hnsw_ef` Qdrant uses at least `limit`. Confirm defaults on Qdrant's official docs for your version.

**Tuning advice:** tune `hnsw_ef` first (query-time only, no re-index). Only touch `m` and `ef_construct` if recall is still lacking. Use `exact=True` as a ground-truth baseline to measure recall, not as your production setting:

```python
exact  = client.query_points(..., search_params=models.SearchParams(exact=True))
approx = client.query_points(..., search_params=models.SearchParams(hnsw_ef=128, exact=False))
recall = len(set(ids(exact)) & set(ids(approx))) / len(ids(exact))
```

### Query retrieval

```
user question
   ↓ ① embed
query vector
   ↓ ② search Qdrant (HNSW, optional filter)
top-k points (id + score + payload)
   ↓ ③ augment: add payload text to the prompt
prompt + context
   ↓ ④ generate: the LLM answers
answer
```

```python
hits = client.query_points(
    collection_name="documents",
    query=query_vector,
    limit=5,
    search_params=models.SearchParams(hnsw_ef=128, exact=False),
    query_filter=models.Filter(must=[
        models.FieldCondition(key="filename", match=models.MatchValue(value="report.pdf"))
    ]),
    with_payload=True,
    score_threshold=0.3,
).points

for h in hits:
    print(h.score, h.payload["content"][:80])
```

- With Cosine, a **higher score means more similar**. A `score_threshold` keeps weak matches away from the LLM.
- **Filters** run against the graph, which is why payload indexes matter.
- **Choosing k** is a trade-off. A common pattern (my suggestion) is to retrieve around 20, then use a reranker to keep the best 3 to 5.
- While embeddings are random, none of this is meaningful: HNSW will faithfully find "closest" vectors that don't mean anything.

---

## 6. Sources

### 6.1 The two tools from your session (read directly)

**Landing AI**
- Parsing Models (DPT-3 Pro vs Verity, snapshots): <https://docs.landing.ai/dpt3/parse-models>
- Documentation index (use this to find the Parse API, credit consumption and file types pages): <https://docs.landing.ai/llms.txt>

**LlamaParse (LlamaIndex)**
- Overview of Parse: <https://developers.llamaindex.ai/llamaparse/parse/>
- OCR and languages: <https://developers.llamaindex.ai/llamaparse/parse/features/ocr-and-languages/>
- Tiers (full comparison): <https://developers.llamaindex.ai/llamaparse/parse/guides/tiers/>
- Configuring Parse (includes Cost Optimizer): <https://developers.llamaindex.ai/llamaparse/parse/guides/configuring-parse/>
- Charts and figures: <https://developers.llamaindex.ai/llamaparse/parse/features/charts-and-figures/>
- Tables: <https://developers.llamaindex.ai/llamaparse/parse/features/tables/>
- Layout and bounding boxes: <https://developers.llamaindex.ai/llamaparse/parse/features/layout-and-bounding-boxes/>
- Pricing: <https://developers.llamaindex.ai/llamaparse/general/pricing/>
- Self-hosting (BYOC): <https://developers.llamaindex.ai/llamaparse/self_hosting/>
- Data retention: <https://developers.llamaindex.ai/llamaparse/general/data_retention/>
- Changelog: <https://developers.llamaindex.ai/llamaparse/changelog/>
- Docs index for programmatic access: <https://developers.llamaindex.ai/llms.txt>
- GitHub organization: <https://github.com/run-llama>

*(I read the Overview and OCR pages in full. The other LlamaParse links above are pages I saw listed in their navigation but did not open, so check them yourself.)*

### 6.2 Docling

- Paper: *Docling Technical Report*, arXiv:2408.09869 — <https://arxiv.org/abs/2408.09869>
- Official documentation: <https://docling-project.github.io/docling/>
- Code: <https://github.com/ds4sd/docling>
- Pipeline options (a third-party mirror of the docs; useful for OCR and picture-description options, but confirm names on the official site): <https://mintlify.com/docling-project/docling/api/options/pipeline-options>

### 6.3 Other document parsers and benchmarks

- *MinerU: An Open-Source Solution for Precise Document Content Extraction*, arXiv:2409.18839 — <https://arxiv.org/abs/2409.18839>
- MinerU code: <https://github.com/opendatalab/MinerU>
- OmniDocBench (benchmark for comparing parsers): <https://github.com/opendatalab/OmniDocBench>
- *olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models*, arXiv:2502.18443 — <https://arxiv.org/abs/2502.18443>
- olmOCR code: <https://github.com/allenai/olmocr>

### 6.4 Visual document retrieval

- *ColPali: Efficient Document Retrieval with Vision Language Models*, arXiv:2407.01449 (ICLR 2025) — <https://arxiv.org/abs/2407.01449>
- ViDoRe benchmark and models: <https://hf.co/vidore>

### 6.5 Qdrant

- Overview: <https://qdrant.tech/documentation/overview/>
- Manage data (points, vectors, payload, collections, indexing, quantization): <https://qdrant.tech/documentation/manage-data/>
- Concepts: <https://qdrant.tech/documentation/concepts/>
- Indexing: <https://qdrant.tech/documentation/manage-data/indexing/>
- Vector search resource optimization (m, ef_construct, hnsw_ef): <https://qdrant.tech/articles/vector-search-resource-optimization/>
- Measuring ANN recall: <https://qdrant.tech/documentation/tutorials-search-engineering/ann-recall/>
- Multivectors for late interaction (course): <https://qdrant.tech/course/essentials/day-5/colbert-multivectors/>
- FastEmbed ColBERT: <https://qdrant.tech/documentation/fastembed/fastembed-colbert/index.md>
- Scaling PDF retrieval with Qdrant: <https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/>

### 6.5b HNSW

- Malkov and Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, arXiv:1603.09320 — <https://arxiv.org/abs/1603.09320>

### 6.6 The project itself

- PROJECT_FLOW.md: <https://github.com/AI-Projects-list/FastAPI-LangGraph-Qdrant/blob/main/PROJECT_FLOW.md>

### 6.7 From memory, not verified here

Double-check these before citing:

- **DocLayNet**, the layout dataset behind Docling's layout analysis (a search result mentioned arXiv:2206.01062, but I didn't open the paper).
- **TableFormer**, the table-structure recognition model.
- **Nougat** (Meta), end-to-end conversion of scientific PDFs to Markdown.

### 6.8 What's mine versus what's sourced

- The page-triage function and thresholds, the four-step pipeline, the payload design, the "retrieve 20, rerank to 5" tip, and the "start with Docling, benchmark 20 pages, use hosted parsers selectively" advice are **my own suggestions**.
- The driving analogy for HNSW is mine.
- The Landing AI and LlamaParse feature statements in section 3 come from the pages linked in 6.1. Statements about what the vendors *claim* are labeled as claims.
- I did **not** verify Persian OCR quality for any tool, pricing for either hosted service, regional availability, or any head-to-head accuracy numbers.

---

## Where to go next

- Walk through the **Query / RAG flow with LangGraph** (embed → retrieve → augment → generate).
- Build the small 20-page benchmark and fill in the session notes in section 3.
- Replace the random embeddings with a real model, then measure recall against `exact=True`.
