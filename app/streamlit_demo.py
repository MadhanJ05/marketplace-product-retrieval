"""
Wholesale Product Search Demo
Faire-inspired UI for hybrid retrieval system
"""

import streamlit as st
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query_understanding import QueryUnderstanding
from retrieval import HybridRetriever
from reranker import CrossEncoderReranker

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Wholesale Product Search",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Faire-inspired styling
# ============================================================================

st.markdown("""
<style>
    /* Global */
    .stApp {
        background-color: #FAFAFA;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #666;
        font-size: 1rem;
    }
    
    /* Product card */
    .product-card {
        background: #fff;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.2s;
        height: 100%;
        border: 1px solid #eee;
        min-height: 140px;
    }
    .product-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Product info */
    .product-rank {
        display: inline-block;
        background: #007D6F;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        text-align: center;
        line-height: 24px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .product-brand {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .product-title {
        font-size: 0.95rem;
        font-weight: 500;
        color: #1a1a1a;
        line-height: 1.4;
        margin-bottom: 0.75rem;
    }
    .product-color {
        display: inline-block;
        background: #f0f0f0;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.7rem;
        color: #666;
    }
    .product-boosted {
        display: inline-block;
        background: #E8F5F3;
        color: #007D6F;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    
    /* Query understanding panel */
    .qu-panel {
        background: #fff;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e8e8e8;
    }
    .qu-label {
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .qu-value {
        font-size: 0.9rem;
        color: #333;
        margin-bottom: 0.75rem;
    }
    .qu-tag {
        display: inline-block;
        background: #E8F5F3;
        color: #007D6F;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
    
    /* Results header */
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #eee;
    }
    .results-count {
        font-size: 0.95rem;
        color: #333;
    }
    .results-latency {
        background: #E8F5F3;
        color: #007D6F;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: #007D6F;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: #006557;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (cached)
# ============================================================================

@st.cache_resource
def load_query_understanding():
    return QueryUnderstanding()

@st.cache_resource
def load_retriever():
    return HybridRetriever()

@st.cache_resource  
def load_reranker():
    return CrossEncoderReranker()

@st.cache_data
def load_products():
    import pandas as pd
    df = pd.read_parquet("data/processed/products.parquet")
    return df.set_index("product_id").to_dict("index")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def truncate(text: str, max_len: int = 60) -> str:
    if not text or len(text) <= max_len:
        return text or ""
    return text[:max_len].rsplit(' ', 1)[0] + "..."

def render_product_card(rank: int, title: str, brand: str, color: str, is_boosted: bool = False):
    color_tag = f'<div class="product-color">{truncate(color, 25)}</div>' if color else ''
    boosted_tag = '<span class="product-boosted">wholesale+</span>' if is_boosted else ''
    
    st.markdown(f"""
    <div class="product-card">
        <div class="product-rank">{rank}</div>
        <div class="product-brand">{truncate(brand, 30) or "Unknown Brand"}</div>
        <div class="product-title">{title or "Product"}</div>
        {color_tag}{boosted_tag}
    </div>
    """, unsafe_allow_html=True)

def render_query_understanding(qu_result: dict):
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown('<div class="qu-label">Intent</div>', unsafe_allow_html=True)
        intent = qu_result.get("intent", "product_search")
        st.markdown(f'<div class="qu-value">{intent}</div>', unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown('<div class="qu-label">Product Type</div>', unsafe_allow_html=True)
        ptype = qu_result.get("product_type", "—")
        st.markdown(f'<div class="qu-value">{ptype}</div>', unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown('<div class="qu-label">Attributes</div>', unsafe_allow_html=True)
        attrs = qu_result.get("attributes", [])
        if attrs:
            attr_html = " ".join([f'<span class="qu-tag">{a}</span>' for a in attrs[:5]])
        else:
            attr_html = '<span style="color:#888">—</span>'
        st.markdown(f'<div class="qu-value">{attr_html}</div>', unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown('<div class="qu-label">Expanded Terms</div>', unsafe_allow_html=True)
        terms = qu_result.get("expanded_terms", [])
        if terms:
            terms_html = " ".join([f'<span class="qu-tag">{t}</span>' for t in terms[:5]])
        else:
            terms_html = '<span style="color:#888">—</span>'
        st.markdown(f'<div class="qu-value">{terms_html}</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Search Settings")
    
    retrieval_method = st.radio(
        "Retrieval Method",
        ["Hybrid (BM25 + FAISS)", "BM25 Only", "FAISS Only"],
        index=0
    )
    
    use_reranker = st.checkbox("Enable CrossEncoder Reranker", value=True)
    
    top_k = st.slider("Number of results", 4, 20, 12, step=4)
    
    show_qu = st.checkbox("Show query analysis", value=True)
    
    show_latency = st.checkbox("Show latency breakdown", value=False)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #888;">
        <strong>Pipeline</strong><br>
        1. Query Understanding (LLM)<br>
        2. BM25 + FAISS Retrieval<br>
        3. Hybrid RRF Fusion<br>
        4. CrossEncoder Reranking<br>
        5. Wholesale Boost
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🏪 Wholesale Product Search</h1>
    <p>AI-powered hybrid search for B2B marketplace</p>
</div>
""", unsafe_allow_html=True)

# Search bar
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    query = st.text_input(
        "Search",
        placeholder="Search products...",
        label_visibility="collapsed"
    )

# Suggested queries
suggestions = ["bulk ceramic mugs", "organic coffee wholesale", "scented candles", "kitchen utensils set", "cotton towels bulk"]

cols = st.columns(len(suggestions) + 2)
for i, sug in enumerate(suggestions):
    with cols[i + 1]:
        if st.button(sug, key=f"sug_{i}"):
            st.session_state["query"] = sug
            st.rerun()

# Check for suggestion click
if "query" in st.session_state and st.session_state["query"]:
    query = st.session_state["query"]
    st.session_state["query"] = ""

# ============================================================================
# SEARCH & RESULTS
# ============================================================================

if query:
    # Load components
    with st.spinner("Loading models..."):
        qu_model = load_query_understanding()
        retriever = load_retriever()
        reranker = load_reranker() if use_reranker else None
        products_dict = load_products()
    
    # WHOLESALE APP: Check if query already has wholesale signals
    wholesale_signals = ["bulk", "wholesale", "case", "pack", "set of", "dozen"]
    has_wholesale_signal = any(signal in query.lower() for signal in wholesale_signals)
    
    # Timing
    timings = {}
    pipeline_status = {}
    
    # Step 1: Query Understanding
    t0 = time.time()
    qu_result = qu_model.analyze(query)
    timings["Query Understanding"] = (time.time() - t0) * 1000
    pipeline_status["1. Query Understanding (LLM)"] = "✅"
    
    # Build search query with expanded terms
    search_query = query
    if qu_result.get("expanded_terms"):
        search_query = f"{query} {' '.join(qu_result['expanded_terms'][:3])}"
    
    # Step 2: Retrieval
    t0 = time.time()
    if retrieval_method == "BM25 Only":
        results = retriever.search_bm25(search_query, top_k=top_k * 2)
        pipeline_status["2. BM25 Retrieval"] = "✅"
        pipeline_status["3. Hybrid RRF Fusion"] = "⏭️ skipped"
    elif retrieval_method == "Dense Only":
        results = retriever.search_faiss(search_query, top_k=top_k * 2)
        pipeline_status["2. FAISS Retrieval"] = "✅"
        pipeline_status["3. Hybrid RRF Fusion"] = "⏭️ skipped"
    else:
        results = retriever.search(search_query, top_k=top_k * 2, method="hybrid")
        pipeline_status["2. BM25 + FAISS Retrieval"] = "✅"
        pipeline_status["3. Hybrid RRF Fusion"] = "✅"
    timings["Retrieval"] = (time.time() - t0) * 1000
    
    # Step 3: Reranking
    if use_reranker and reranker and results:
        t0 = time.time()
        
        # Add product titles for reranking
        for r in results:
            pid = r["product_id"]
            p = products_dict.get(pid, {})
            r["product_title"] = p.get("product_title", "")
        
        # WHOLESALE APP: Always apply bulk boost (this is a B2B marketplace)
        # Modify query to trigger bulk boost in reranker
        rerank_query = query if has_wholesale_signal else f"{query} bulk"
        
        # Rerank with wholesale-enhanced query
        results = reranker.rerank(rerank_query, results, top_k=top_k, text_field="product_title")
        timings["Reranking"] = (time.time() - t0) * 1000
        pipeline_status["4. CrossEncoder Reranking"] = "✅"
        
        # Check if bulk boost was applied
        boosted_count = sum(1 for r in results if r.get("bulk_boost", 0) > 0)
        pipeline_status["5. Wholesale Boost"] = f"✅ ({boosted_count} products boosted)"
    else:
        results = results[:top_k]
        pipeline_status["4. CrossEncoder Reranking"] = "⏭️ disabled"
        pipeline_status["5. Wholesale Boost"] = "⏭️ skipped"
    
    total_time = sum(timings.values())
    
    # Show Query Understanding
    if show_qu:
        with st.expander("🔍 Query Analysis", expanded=True):
            render_query_understanding(qu_result)
    
    # Results header
    st.markdown(f"""
    <div class="results-header">
        <span class="results-count">Showing {len(results)} results for "<strong>{query}</strong>"</span>
        <span class="results-latency">⚡ {total_time:.0f}ms</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Latency breakdown
    if show_latency:
        with st.expander("📊 Latency Breakdown"):
            cols = st.columns(len(timings))
            for i, (stage, ms) in enumerate(timings.items()):
                with cols[i]:
                    st.metric(stage, f"{ms:.0f}ms")
    
    # Pipeline status
    with st.expander("🔄 Pipeline Status", expanded=False):
        for step, status in pipeline_status.items():
            st.write(f"{status} {step}")
    
    # Product grid
    cols_per_row = 4
    for row_start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(results):
                result = results[idx]
                pid = result["product_id"]
                product = products_dict.get(pid, {})
                is_boosted = result.get("bulk_boost", 0) > 0
                
                with col:
                    render_product_card(
                        rank=idx + 1,
                        title=product.get("product_title", ""),
                        brand=product.get("product_brand", ""),
                        color=product.get("product_color", ""),
                        is_boosted=is_boosted
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #888;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
        <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Search for wholesale products</div>
        <div style="font-size: 0.9rem;">Try one of the suggestions above to get started</div>
    </div>
    """, unsafe_allow_html=True)