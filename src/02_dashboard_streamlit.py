from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# =========================
# CONFIG
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "vendas_enriched.csv"

COL_DATE = "DataVenda"
COL_VEND_ID = "CodVendedor"
COL_VEND_NOME = "NomeVendedor"
COL_CLI_ID = "CodCliente"
COL_CLI_NOME = "NomeCliente"
COL_PROD_ID = "CodProduto"
COL_PROD_DESC = "DescricaoProdutos"
COL_FILIAL = "CodFilial"
COL_UF = "EstadoCliente"
COL_CIDADE = "CidadeCliente"
COL_VALOR = "ValorLiquido"
COL_PEDIDO = "Codigo"


# =========================
# HELPERS
# =========================
def br_region_from_uf(uf: str) -> str:
    if not isinstance(uf, str):
        return "Indefinida"
    uf = uf.strip().upper()

    norte = {"AC", "AP", "AM", "PA", "RO", "RR", "TO"}
    nordeste = {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"}
    centro_oeste = {"DF", "GO", "MT", "MS"}
    sudeste = {"ES", "MG", "RJ", "SP"}
    sul = {"PR", "RS", "SC"}

    if uf in norte:
        return "Norte"
    if uf in nordeste:
        return "Nordeste"
    if uf in centro_oeste:
        return "Centro-Oeste"
    if uf in sudeste:
        return "Sudeste"
    if uf in sul:
        return "Sul"
    return "Indefinida"


def br_money(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def infer_is_cents(series: pd.Series) -> bool:
    """
    Heurística para inferir centavos:
    - muitos valores inteiros (ou muito próximos de inteiro)
    - mediana alta indicando centavos
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False

    frac = (np.abs(s - np.round(s)) < 1e-6).mean()
    med = float(s.median())
    return (frac > 0.90) and (med > 10_000)


def apply_brl_currency_axis(fig, axis: str = "y", title: str | None = None):
    """
    Formata o eixo numérico como moeda BRL (R$ 1.234,56) para Plotly.
    axis: "x" ou "y" (qual eixo está com o valor/faturamento).
    """
    fig.update_layout(separators=",.")  # decimal "," e milhar "."
    if axis == "y":
        fig.update_yaxes(tickprefix="R$ ", tickformat=",.2f", title=title)
    else:
        fig.update_xaxes(tickprefix="R$ ", tickformat=",.2f", title=title)
    return fig


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    # Data
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df[df[COL_DATE].notna()].copy()

    # Valor
    df[COL_VALOR] = pd.to_numeric(df[COL_VALOR], errors="coerce").fillna(0.0)

    # Se parecer centavos -> reais
    if infer_is_cents(df[COL_VALOR]):
        df[COL_VALOR] = df[COL_VALOR] / 100.0

    # Derivadas de data
    df["ano"] = df[COL_DATE].dt.year.astype(int)
    df["mes"] = df[COL_DATE].dt.month.astype(int)
    df["dia"] = df[COL_DATE].dt.day.astype(int)

    df["ano_mes"] = df[COL_DATE].dt.to_period("M").astype(str)

    iso = df[COL_DATE].dt.isocalendar()
    df["semana"] = iso.week.astype(int)
    df["ano_semana"] = iso.year.astype(int).astype(str) + "-W" + df["semana"].astype(str).str.zfill(2)

    # Dia da semana
    df["dow_num"] = df[COL_DATE].dt.dayofweek.astype(int)  # seg=0..dom=6
    dow_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
    df["dia_semana"] = df["dow_num"].map(dow_names)

    # Região
    if COL_UF in df.columns:
        df["regiao"] = df[COL_UF].apply(br_region_from_uf)
    else:
        df["regiao"] = "Indefinida"

    # Strings (limpa)
    for c in [COL_VEND_NOME, COL_CLI_NOME, COL_PROD_DESC, COL_UF, COL_CIDADE]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def apply_filters(df: pd.DataFrame, years, months, filiais, regioes) -> pd.DataFrame:
    out = df.copy()
    if years:
        out = out[out["ano"].isin(years)]
    if months:
        out = out[out["mes"].isin(months)]
    if filiais and COL_FILIAL in out.columns:
        out = out[out[COL_FILIAL].isin(filiais)]
    if regioes and "regiao" in out.columns:
        out = out[out["regiao"].isin(regioes)]
    return out


def agg_rank(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    g = (
        df.groupby(by_cols, as_index=False)
        .agg(
            faturamento=(COL_VALOR, "sum"),
            pedidos=(COL_PEDIDO, "nunique"),
        )
    )
    g["faturamento"] = g["faturamento"].round(2)
    return g


def add_share_percent(rank_df: pd.DataFrame) -> pd.DataFrame:
    out = rank_df.copy()
    total = float(out["faturamento"].sum())
    out["share_%"] = np.where(total > 0, (out["faturamento"] / total) * 100, 0.0).round(2)
    return out


def top_bottom(df_rank: pd.DataFrame, n: int):
    top = df_rank.sort_values("faturamento", ascending=False).head(n)
    bottom = df_rank.sort_values("faturamento", ascending=True).head(n)
    return top, bottom


def share_top_n(df_rank: pd.DataFrame, n: int) -> float:
    total = df_rank["faturamento"].sum()
    top_sum = df_rank.sort_values("faturamento", ascending=False).head(n)["faturamento"].sum()
    return float((top_sum / total * 100) if total > 0 else 0.0)


def agg_timeseries(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    if grain == "D":
        key = df[COL_DATE].dt.date.astype(str)
        label = "dia"
    elif grain == "W":
        key = df["ano_semana"]
        label = "ano_semana"
    else:
        key = df["ano_mes"]
        label = "ano_mes"

    ts = (
        df.assign(_k=key)
        .groupby("_k", as_index=False)
        .agg(faturamento=(COL_VALOR, "sum"), pedidos=(COL_PEDIDO, "nunique"))
        .rename(columns={"_k": label})
    )
    ts["faturamento"] = ts["faturamento"].round(2)

    # ordenar
    if label == "dia":
        ts[label] = pd.to_datetime(ts[label])
        ts = ts.sort_values(label)
    elif label == "ano_mes":
        ts["_ord"] = pd.to_datetime(ts[label] + "-01")
        ts = ts.sort_values("_ord").drop(columns=["_ord"])
    else:
        parts = ts[label].str.split("-W", expand=True)
        ts["_y"] = parts[0].astype(int)
        ts["_w"] = parts[1].astype(int)
        ts = ts.sort_values(["_y", "_w"]).drop(columns=["_y", "_w"])

    return ts


def month_name_pt(m: int) -> str:
    names = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
    return names.get(int(m), str(m))


def get_last_month_info(df: pd.DataFrame):
    if df.empty:
        return None, True, None
    last_date = df[COL_DATE].max()
    m_last = str(last_date.to_period("M"))
    month_end = last_date.to_period("M").to_timestamp(how="end").normalize()
    is_complete = last_date.normalize() >= month_end
    return m_last, is_complete, last_date


def compute_mom_growth(df: pd.DataFrame, by_cols: list[str], exclude_incomplete_last_month: bool = True) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    months = sorted(df["ano_mes"].unique().tolist(), key=lambda x: pd.to_datetime(str(x) + "-01"))

    if len(months) < 2:
        return pd.DataFrame()

    m_last, is_complete, _ = get_last_month_info(df)

    use_months = months.copy()
    if exclude_incomplete_last_month and (m_last is not None) and (m_last == months[-1]) and (not is_complete):
        use_months = months[:-1]

    if len(use_months) < 2:
        return pd.DataFrame()

    m_prev, m_curr = use_months[-2], use_months[-1]

    prev = (
        df[df["ano_mes"] == m_prev]
        .groupby(by_cols, as_index=False)
        .agg(fat_prev=(COL_VALOR, "sum"), ped_prev=(COL_PEDIDO, "nunique"))
    )
    curr = (
        df[df["ano_mes"] == m_curr]
        .groupby(by_cols, as_index=False)
        .agg(fat_curr=(COL_VALOR, "sum"), ped_curr=(COL_PEDIDO, "nunique"))
    )

    out = curr.merge(prev, on=by_cols, how="outer").fillna(0)
    out["delta_fat"] = out["fat_curr"] - out["fat_prev"]
    out["delta_%"] = np.where(out["fat_prev"] > 0, (out["delta_fat"] / out["fat_prev"]) * 100, np.nan)

    out["fat_curr"] = out["fat_curr"].round(2)
    out["fat_prev"] = out["fat_prev"].round(2)
    out["delta_fat"] = out["delta_fat"].round(2)
    out["delta_%"] = out["delta_%"].round(2)

    out["_m_prev"] = m_prev
    out["_m_curr"] = m_curr
    return out


def heatmap_month_dow(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    pivot = (
        df.groupby(["mes", "dia_semana"], as_index=False)
        .agg(faturamento=(COL_VALOR, "sum"))
    )
    pivot["mes_nome"] = pivot["mes"].apply(month_name_pt)

    dow_order = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    pivot["dia_semana"] = pd.Categorical(pivot["dia_semana"], categories=dow_order, ordered=True)

    return pivot


def pareto_chart(df_rank: pd.DataFrame, name_col: str, n: int, title: str):
    if df_rank.empty:
        return None

    d = df_rank.sort_values("faturamento", ascending=False).head(n).copy()
    d["cum_fat"] = d["faturamento"].cumsum()
    total = float(d["faturamento"].sum())
    d["cum_%"] = np.where(total > 0, (d["cum_fat"] / total) * 100, 0.0)

    d = d.iloc[::-1].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=d["faturamento"],
            y=d[name_col].astype(str),
            orientation="h",
            name="Faturamento",
            hovertemplate="<b>%{y}</b><br>Faturamento: R$ %{x:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["cum_%"],
            y=d[name_col].astype(str),
            mode="lines+markers",
            name="% acumulado (Top N)",
            xaxis="x2",
            hovertemplate="<b>%{y}</b><br>% acumulado: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Faturamento (R$)", showgrid=True),
        xaxis2=dict(
            title="% acumulado (Top N)",
            overlaying="x",
            side="top",
            range=[0, 100],
            ticksuffix="%",
            showgrid=False,
        ),
        yaxis=dict(title=""),
        separators=",.",
    )

    # eixo do faturamento (xaxis)
    fig.update_xaxes(tickprefix="R$ ", tickformat=",.2f")

    return fig


def bar_top_with_share(df_top: pd.DataFrame, name_col: str, title: str):
    if df_top.empty:
        return None
    fig = px.bar(
        df_top.sort_values("faturamento"),
        x="faturamento",
        y=name_col,
        orientation="h",
        hover_data={"pedidos": True, "share_%": True, "faturamento": ":,.2f"},
        labels={name_col: "Nome", "faturamento": "Faturamento", "pedidos": "Pedidos", "share_%": "% do total"},
        title=title,
    )
    apply_brl_currency_axis(fig, axis="x", title="Faturamento (R$)")
    return fig


# =========================
# NOVO: M/M seguindo o mês selecionado
# =========================
def month_period(m) -> pd.Period:
    if m is None:
        raise ValueError("month_period recebeu None")

    if isinstance(m, pd.Period):
        return m.asfreq("M")

    if isinstance(m, pd.Timestamp):
        return m.to_period("M")

    try:
        return pd.to_datetime(m).to_period("M")
    except Exception:
        return pd.Period(str(m), freq="M")


def mom_for_fixed_month(
    df_base: pd.DataFrame,
    by_cols: list[str],
    m_curr: str,
    exclude_incomplete_last_month: bool = True,
):
    if (df_base is None) or df_base.empty:
        return pd.DataFrame(), None, None, "Sem dados no recorte base do M/M.", None

    if not m_curr:
        return pd.DataFrame(), None, None, "Sem mês base para M/M (verifique filtros).", None

    p_curr = month_period(m_curr)
    p_prev = p_curr - 1
    m_curr_s = str(p_curr)
    m_prev_s = str(p_prev)

    if exclude_incomplete_last_month:
        m_last, is_complete, last_date = get_last_month_info(df_base)
        if (m_last == m_curr_s) and (not is_complete):
            return (
                pd.DataFrame(),
                m_prev_s,
                m_curr_s,
                f"O mês atual do M/M ({m_curr_s}) está incompleto (última data: {last_date.date()}). "
                "Desmarque 'Excluir mês incompleto no M/M' ou escolha um mês completo.",
                None,
            )

    df_2m = df_base[df_base["ano_mes"].isin([m_prev_s, m_curr_s])].copy()

    if df_2m["ano_mes"].nunique() < 2:
        return (
            pd.DataFrame(),
            m_prev_s,
            m_curr_s,
            f"Não há dados suficientes para comparar {m_prev_s} → {m_curr_s} dentro do recorte (ano/filial/região).",
            None,
        )

    mom = compute_mom_growth(df_2m, by_cols, exclude_incomplete_last_month=False)
    if mom.empty:
        return pd.DataFrame(), m_prev_s, m_curr_s, "Não foi possível calcular crescimento com o recorte atual.", None

    return mom, m_prev_s, m_curr_s, None, None


# =========================
# UI
# =========================
st.set_page_config(page_title="Dashboard de Vendas", layout="wide")

st.title("📊 Dashboard de Vendas")
st.caption("Visão geral + tops + geografia + tendência + insights (concentração, crescimento e heatmap).")

df = load_data(DATA_PATH)

# Sidebar filtros
st.sidebar.header("Filtros")

years_all = sorted(df["ano"].unique().tolist())
months_all = list(range(1, 13))

years = st.sidebar.multiselect("Ano", options=years_all, default=years_all[-1:] if years_all else [])
months = st.sidebar.multiselect("Mês", options=months_all, default=months_all)

filiais = []
if COL_FILIAL in df.columns:
    filiais_all = sorted(df[COL_FILIAL].dropna().unique().tolist())
    filiais = st.sidebar.multiselect("Filial", options=filiais_all, default=filiais_all)

regioes_all = sorted(df["regiao"].dropna().unique().tolist())
regioes = st.sidebar.multiselect("Região", options=regioes_all, default=regioes_all)

top_n = st.sidebar.slider("Top N", min_value=5, max_value=30, value=10)

grain = st.sidebar.selectbox("Granularidade da tendência", ["Mês", "Semana", "Dia"], index=0)
grain_map = {"Dia": "D", "Semana": "W", "Mês": "M"}
grain_code = grain_map[grain]

exclude_incomplete_mom = st.sidebar.checkbox("Excluir mês incompleto no M/M (recomendado)", value=True)
show_bottom = st.sidebar.checkbox("Mostrar Bottom (menores)", value=False)

# filtros aplicados (recorte principal)
df_f = apply_filters(df, years, months, filiais, regioes)

# BASE DO M/M (sem filtrar mês)
df_base_mom = apply_filters(df, years, [], filiais, regioes)

mom_months_available = sorted(
    df_base_mom["ano_mes"].unique().tolist(),
    key=lambda x: pd.to_datetime(str(x) + "-01")
)

mom_curr_month = None
if mom_months_available:
    if years and months:
        y = max(years)
        m = max(months)
        candidate = f"{y}-{int(m):02d}"
        if candidate in mom_months_available:
            mom_curr_month = candidate

    if mom_curr_month is None:
        mom_curr_month = mom_months_available[-1]

# aviso mês incompleto (no recorte principal)
m_last, is_complete, last_date = get_last_month_info(df_f)
if (m_last is not None) and (not is_complete):
    st.warning(
        f"⚠️ Atenção: o último mês ({m_last}) está incompleto. Última data no recorte: {last_date.date()}. "
        f"Isso pode distorcer tendência e M/M. (Opção no menu lateral)"
    )

# KPIs
c1, c2, c3, c4 = st.columns(4)
fat_total = float(df_f[COL_VALOR].sum())
ped_total = int(df_f[COL_PEDIDO].nunique())
ticket = fat_total / max(1, ped_total)
clientes_unicos = int(df_f[COL_CLI_ID].nunique()) if COL_CLI_ID in df_f.columns else 0

c1.metric("Faturamento (R$)", br_money(fat_total))
c2.metric("Pedidos", ped_total)
c3.metric("Ticket médio (R$)", br_money(ticket))
c4.metric("Clientes únicos", clientes_unicos)

# Export (opcional)
with st.sidebar.expander("Exportar (opcional)", expanded=False):
    csv = df_f.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Baixar CSV do recorte atual",
        data=csv,
        file_name="vendas_filtradas.csv",
        mime="text/csv"
    )

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📌 Visão Geral", "🏆 Vendedores", "👤 Clientes", "🛒 Produtos", "🗺️ Geografia", "💡 Insights"]
)

# =========================
# TAB 1: VISÃO GERAL
# =========================
with tab1:
    st.subheader(f"📈 Tendência de faturamento por {grain.lower()}")
    ts = agg_timeseries(df_f, grain_code)
    x_col = "ano_mes" if grain_code == "M" else ("ano_semana" if grain_code == "W" else "dia")

    fig = px.line(
        ts,
        x=x_col,
        y="faturamento",
        markers=True,
        hover_data={"pedidos": True, "faturamento": ":,.2f"},
        labels={x_col: "Período", "faturamento": "Faturamento", "pedidos": "Pedidos"},
        title=f"Tendência de faturamento por {grain.lower()}",
    )
    apply_brl_currency_axis(fig, axis="y", title="Faturamento (R$)")
    st.plotly_chart(fig, use_container_width=True, key=f"ts_{grain_code}_{len(ts)}")

    st.divider()

    st.subheader("🗺️ Participação no faturamento por Região (com Pedidos no hover)")
    reg = (
        df_f.groupby("regiao", as_index=False)
        .agg(faturamento=(COL_VALOR, "sum"), pedidos=(COL_PEDIDO, "nunique"))
        .sort_values("faturamento", ascending=False)
    )
    reg["faturamento"] = reg["faturamento"].round(2)
    reg["pedidos"] = reg["pedidos"].fillna(0).astype(int)

    fig_pie = px.pie(reg, names="regiao", values="faturamento", title="Participação no faturamento por Região")
    fig_pie.update_layout(separators=",.")
    fig_pie.update_traces(
        textinfo="percent+label",
        hovertemplate=(
            "<b>Região</b>: %{label}<br>"
            "<b>Faturamento</b>: R$ %{value:,.2f}<br>"
            "<b>Participação</b>: %{percent}<br>"
            "<b>Pedidos</b>: %{customdata[0]}<extra></extra>"
        ),
        customdata=reg[["pedidos"]].to_numpy()
    )
    st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_reg_{len(reg)}")

    st.divider()

    st.subheader("🌦️ Sazonalidade (opcional)")
    if not HAS_STATSMODELS:
        st.info("Para decomposição de sazonalidade, instale statsmodels.")
    else:
        st.caption("Melhor com série mensal (>= 24 meses).")
        if grain_code != "M":
            st.warning("Mude a granularidade para 'Mês' para uma decomposição mais estável.")
        else:
            ts2 = ts.copy()
            ts2["_d"] = pd.to_datetime(ts2["ano_mes"] + "-01")
            ts2 = ts2.sort_values("_d").set_index("_d")

            if len(ts2) >= 24:
                result = seasonal_decompose(ts2["faturamento"], model="additive", period=12)
                comp = pd.DataFrame({
                    "observado": result.observed,
                    "tendencia": result.trend,
                    "sazonal": result.seasonal,
                    "residuo": result.resid
                }).reset_index()

                fig_obs = px.line(comp, x="_d", y="observado", title="Série observada")
                apply_brl_currency_axis(fig_obs, axis="y", title="Faturamento (R$)")
                st.plotly_chart(fig_obs, use_container_width=True, key="decomp_obs")

                fig_tr = px.line(comp, x="_d", y="tendencia", title="Tendência")
                apply_brl_currency_axis(fig_tr, axis="y", title="Faturamento (R$)")
                st.plotly_chart(fig_tr, use_container_width=True, key="decomp_trend")

                fig_se = px.line(comp, x="_d", y="sazonal", title="Sazonalidade")
                apply_brl_currency_axis(fig_se, axis="y", title="Faturamento (R$)")
                st.plotly_chart(fig_se, use_container_width=True, key="decomp_season")
            else:
                st.warning("Poucos meses para decompor (ideal ≥ 24 meses).")


# =========================
# TAB 2: VENDEDORES
# =========================
with tab2:
    vend_rank = add_share_percent(agg_rank(df_f, [COL_VEND_ID, COL_VEND_NOME]))
    st.subheader("🏆 Vendedores")

    c1, c2 = st.columns(2)
    c1.metric(f"Participação do Top {top_n}", f"{share_top_n(vend_rank, top_n):.2f}%")
    c2.metric("Qtd. vendedores no recorte", int(vend_rank.shape[0]))

    st.divider()

    st.caption("📌 Pareto (Concentração) — Vendedores (Top N)")
    fig_par = pareto_chart(vend_rank, COL_VEND_NOME, top_n, title=f"Pareto — Vendedores (Top {top_n})")
    if fig_par is not None:
        st.plotly_chart(fig_par, use_container_width=True, key=f"pareto_vend_{top_n}_{len(vend_rank)}")

    st.divider()

    top_v, bottom_v = top_bottom(vend_rank, top_n)

    st.caption(f"Top {top_n} Vendedores (com % do total no hover)")
    fig = bar_top_with_share(top_v, COL_VEND_NOME, title=f"Top {top_n} Vendedores — Faturamento")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"top_vend_{top_n}_{len(top_v)}")

    if show_bottom:
        st.caption(f"Bottom {top_n} Vendedores (menor faturamento)")
        figb = px.bar(
            bottom_v.sort_values("faturamento"),
            x="faturamento",
            y=COL_VEND_NOME,
            orientation="h",
            hover_data={"pedidos": True, "share_%": True, "faturamento": ":,.2f"},
            labels={COL_VEND_NOME: "Vendedor", "faturamento": "Faturamento", "pedidos": "Pedidos", "share_%": "% do total"},
        )
        apply_brl_currency_axis(figb, axis="x", title="Faturamento (R$)")
        st.plotly_chart(figb, use_container_width=True, key=f"bottom_vend_{top_n}_{len(bottom_v)}")

    st.divider()

    st.subheader("📈 Crescimento (Mês atual vs Mês anterior)")
    mom, m_prev, m_curr, err, warn = mom_for_fixed_month(
        df_base_mom,
        [COL_VEND_ID, COL_VEND_NOME],
        mom_curr_month,
        exclude_incomplete_last_month=exclude_incomplete_mom,
    )

    if warn:
        st.warning(warn)
    if err:
        st.info(err)
    else:
        st.caption(f"Comparação (M/M): {m_prev} → {m_curr}")

        col_up, col_down = st.columns(2)

        with col_up:
            st.caption(f"Top {top_n} - Maior alta (Δ faturamento)")
            top_up = mom.sort_values("delta_fat", ascending=False).head(top_n)
            fig_up = px.bar(
                top_up.sort_values("delta_fat"),
                x="delta_fat",
                y=COL_VEND_NOME,
                orientation="h",
                hover_data={"fat_prev": ":,.2f", "fat_curr": ":,.2f", "delta_%": True},
                labels={"delta_fat": "Δ Faturamento", COL_VEND_NOME: "Vendedor"},
            )
            apply_brl_currency_axis(fig_up, axis="x", title="Δ Faturamento (R$)")
            st.plotly_chart(fig_up, use_container_width=True, key=f"mom_vend_up_{m_prev}_{m_curr}_{top_n}")

        with col_down:
            st.caption(f"Top {top_n} - Maior queda (Δ faturamento)")
            top_down = mom.sort_values("delta_fat", ascending=True).head(top_n)
            fig_down = px.bar(
                top_down.sort_values("delta_fat"),
                x="delta_fat",
                y=COL_VEND_NOME,
                orientation="h",
                hover_data={"fat_prev": ":,.2f", "fat_curr": ":,.2f", "delta_%": True},
                labels={"delta_fat": "Δ Faturamento", COL_VEND_NOME: "Vendedor"},
            )
            apply_brl_currency_axis(fig_down, axis="x", title="Δ Faturamento (R$)")
            st.plotly_chart(fig_down, use_container_width=True, key=f"mom_vend_down_{m_prev}_{m_curr}_{top_n}")


# =========================
# TAB 3: CLIENTES
# =========================
with tab3:
    cli_rank = add_share_percent(agg_rank(df_f, [COL_CLI_ID, COL_CLI_NOME]))
    st.subheader("👤 Clientes")

    c1, c2 = st.columns(2)
    c1.metric(f"Participação do Top {top_n}", f"{share_top_n(cli_rank, top_n):.2f}%")
    c2.metric("Qtd. clientes no recorte", int(cli_rank.shape[0]))

    st.divider()

    st.caption("📌 Pareto (Concentração) — Clientes (Top N)")
    fig_par = pareto_chart(cli_rank, COL_CLI_NOME, top_n, title=f"Pareto — Clientes (Top {top_n})")
    if fig_par is not None:
        st.plotly_chart(fig_par, use_container_width=True, key=f"pareto_cli_{top_n}_{len(cli_rank)}")

    st.divider()

    top_c, bottom_c = top_bottom(cli_rank, top_n)

    st.caption(f"Top {top_n} Clientes (com % do total no hover)")
    fig = bar_top_with_share(top_c, COL_CLI_NOME, title=f"Top {top_n} Clientes — Faturamento")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"top_cli_{top_n}_{len(top_c)}")

    if show_bottom:
        st.caption(f"Bottom {top_n} Clientes (menor faturamento)")
        figb = px.bar(
            bottom_c.sort_values("faturamento"),
            x="faturamento",
            y=COL_CLI_NOME,
            orientation="h",
            hover_data={"pedidos": True, "share_%": True, "faturamento": ":,.2f"},
            labels={COL_CLI_NOME: "Cliente", "faturamento": "Faturamento", "pedidos": "Pedidos", "share_%": "% do total"},
        )
        apply_brl_currency_axis(figb, axis="x", title="Faturamento (R$)")
        st.plotly_chart(figb, use_container_width=True, key=f"bottom_cli_{top_n}_{len(bottom_c)}")

    st.divider()

    st.subheader("📈 Crescimento (Mês atual vs Mês anterior)")
    mom, m_prev, m_curr, err, warn = mom_for_fixed_month(
        df_base_mom,
        [COL_CLI_ID, COL_CLI_NOME],
        mom_curr_month,
        exclude_incomplete_last_month=exclude_incomplete_mom,
    )

    if warn:
        st.warning(warn)
    if err:
        st.info(err)
    else:
        st.caption(f"Comparação (M/M): {m_prev} → {m_curr}")

        col_up, col_down = st.columns(2)

        with col_up:
            st.caption(f"Top {top_n} - Maior alta (Δ faturamento)")
            top_up = mom.sort_values("delta_fat", ascending=False).head(top_n)
            fig_up = px.bar(
                top_up.sort_values("delta_fat"),
                x="delta_fat",
                y=COL_CLI_NOME,
                orientation="h",
                hover_data={"fat_prev": ":,.2f", "fat_curr": ":,.2f", "delta_%": True},
            )
            apply_brl_currency_axis(fig_up, axis="x", title="Δ Faturamento (R$)")
            st.plotly_chart(fig_up, use_container_width=True, key=f"mom_cli_up_{m_prev}_{m_curr}_{top_n}")

        with col_down:
            st.caption(f"Top {top_n} - Maior queda (Δ faturamento)")
            top_down = mom.sort_values("delta_fat", ascending=True).head(top_n)
            fig_down = px.bar(
                top_down.sort_values("delta_fat"),
                x="delta_fat",
                y=COL_CLI_NOME,
                orientation="h",
                hover_data={"fat_prev": ":,.2f", "fat_curr": ":,.2f", "delta_%": True},
            )
            apply_brl_currency_axis(fig_down, axis="x", title="Δ Faturamento (R$)")
            st.plotly_chart(fig_down, use_container_width=True, key=f"mom_cli_down_{m_prev}_{m_curr}_{top_n}")


# =========================
# TAB 4: PRODUTOS
# =========================
with tab4:
    prod_rank = add_share_percent(agg_rank(df_f, [COL_PROD_ID, COL_PROD_DESC]))
    st.subheader("🛒 Produtos")

    c1, c2 = st.columns(2)
    c1.metric(f"Participação do Top {top_n}", f"{share_top_n(prod_rank, top_n):.2f}%")
    c2.metric("Qtd. produtos no recorte", int(prod_rank.shape[0]))

    st.divider()

    st.caption("📌 Pareto (Concentração) — Produtos (Top N)")
    fig_par = pareto_chart(prod_rank, COL_PROD_DESC, top_n, title=f"Pareto — Produtos (Top {top_n})")
    if fig_par is not None:
        st.plotly_chart(fig_par, use_container_width=True, key=f"pareto_prod_{top_n}_{len(prod_rank)}")

    st.divider()

    top_p, bottom_p = top_bottom(prod_rank, top_n)

    st.caption(f"Top {top_n} Produtos (com % do total no hover)")
    fig = bar_top_with_share(top_p, COL_PROD_DESC, title=f"Top {top_n} Produtos — Faturamento")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"top_prod_{top_n}_{len(top_p)}")

    if show_bottom:
        st.caption(f"Bottom {top_n} Produtos (menor faturamento)")
        figb = px.bar(
            bottom_p.sort_values("faturamento"),
            x="faturamento",
            y=COL_PROD_DESC,
            orientation="h",
            hover_data={"pedidos": True, "share_%": True, "faturamento": ":,.2f"},
            labels={COL_PROD_DESC: "Produto", "faturamento": "Faturamento", "pedidos": "Pedidos", "share_%": "% do total"},
        )
        apply_brl_currency_axis(figb, axis="x", title="Faturamento (R$)")
        st.plotly_chart(figb, use_container_width=True, key=f"bottom_prod_{top_n}_{len(bottom_p)}")

    st.divider()

    st.subheader("📈 Crescimento (Mês atual vs Mês anterior)")
    mom, m_prev, m_curr, err, warn = mom_for_fixed_month(
        df_base_mom,
        [COL_PROD_ID, COL_PROD_DESC],
        mom_curr_month,
        exclude_incomplete_last_month=exclude_incomplete_mom,
    )

    if warn:
        st.warning(warn)
    if err:
        st.info(err)
    else:
        st.caption(f"Comparação (M/M): {m_prev} → {m_curr}")

        col_up, col_down = st.columns(2)

        with col_up:
            st.caption(f"Top {top_n} - Maior alta (Δ faturamento)")
            top_up = mom.sort_values("delta_fat", ascending=False).head(top_n)
            fig_up = px.bar(
                top_up.sort_values("delta_fat"),
                x="delta_fat",
                y=COL_PROD_DESC,
                orientation="h",
                hover_data={"fat_prev": ":,.2f", "fat_curr": ":,.2f", "delta_%": True},
            )
            apply_brl_currency_axis(fig_up, axis="x", title="Δ Faturamento (R$)")
            st.plotly_chart(fig_up, use_container_width=True, key=f"mom_prod_up_{m_prev}_{m_curr}_{top_n}")

        with col_down:
            st.caption(f"Top {top_n} - Maior queda (Δ faturamento)")
            top_down = mom.sort_values("delta_fat", ascending=True).head(top_n)
            fig_down = px.bar(
                top_down.sort_values("delta_fat"),
                x="delta_fat",
                y=COL_PROD_DESC,
                orientation="h",
                hover_data={"fat_prev": ":,.2f", "fat_curr": ":,.2f", "delta_%": True},
            )
            apply_brl_currency_axis(fig_down, axis="x", title="Δ Faturamento (R$)")
            st.plotly_chart(fig_down, use_container_width=True, key=f"mom_prod_down_{m_prev}_{m_curr}_{top_n}")


# =========================
# TAB 5: GEOGRAFIA
# =========================
with tab5:
    st.subheader("🗺️ Geografia")

    if COL_UF in df_f.columns:
        geo = (
            df_f.groupby(["regiao", COL_UF], as_index=False)
            .agg(faturamento=(COL_VALOR, "sum"), pedidos=(COL_PEDIDO, "nunique"))
            .sort_values("faturamento", ascending=False)
        )
        geo["faturamento"] = geo["faturamento"].round(2)

        fig_uf = px.bar(
            geo.head(top_n).sort_values("faturamento"),
            x="faturamento",
            y=COL_UF,
            orientation="h",
            color="regiao",
            hover_data={"pedidos": True, "faturamento": ":,.2f"},
            labels={COL_UF: "UF", "regiao": "Região", "faturamento": "Faturamento", "pedidos": "Pedidos"},
            title=f"Top {top_n} UF por faturamento"
        )
        apply_brl_currency_axis(fig_uf, axis="x", title="Faturamento (R$)")
        st.plotly_chart(fig_uf, use_container_width=True, key=f"geo_uf_{top_n}_{len(geo)}")
    else:
        st.info("Coluna de UF não encontrada no CSV.")

    st.divider()

    if COL_CIDADE in df_f.columns and COL_UF in df_f.columns:
        st.subheader(f"🏙️ Top {top_n} Cidades por faturamento")
        city = (
            df_f.groupby([COL_CIDADE, COL_UF, "regiao"], as_index=False)
            .agg(faturamento=(COL_VALOR, "sum"), pedidos=(COL_PEDIDO, "nunique"))
            .sort_values("faturamento", ascending=False)
        )
        city["faturamento"] = city["faturamento"].round(2)

        fig_city = px.bar(
            city.head(top_n).sort_values("faturamento"),
            x="faturamento",
            y=COL_CIDADE,
            orientation="h",
            hover_data={"pedidos": True, COL_UF: True, "regiao": True, "faturamento": ":,.2f"},
            labels={COL_CIDADE: "Cidade", "faturamento": "Faturamento", "pedidos": "Pedidos", COL_UF: "UF", "regiao": "Região"},
            title=f"Top {top_n} Cidades por faturamento"
        )
        apply_brl_currency_axis(fig_city, axis="x", title="Faturamento (R$)")
        st.plotly_chart(fig_city, use_container_width=True, key=f"geo_city_{top_n}_{len(city)}")

        st.divider()
        st.subheader("🌳 Treemap (Região → UF → Cidade)")
        fig_tree = px.treemap(
            city,
            path=["regiao", COL_UF, COL_CIDADE],
            values="faturamento",
            hover_data={"pedidos": True, "faturamento": ":,.2f"},
        )
        fig_tree.update_layout(separators=",.")
        st.plotly_chart(fig_tree, use_container_width=True, key=f"geo_tree_{len(city)}")
    else:
        st.info("Colunas de Cidade/UF não encontradas no CSV.")


# =========================
# TAB 6: INSIGHTS
# =========================
with tab6:
    st.subheader("💡 Insights rápidos")

    vend_rank2 = agg_rank(df_f, [COL_VEND_ID, COL_VEND_NOME])
    cli_rank2 = agg_rank(df_f, [COL_CLI_ID, COL_CLI_NOME])
    prod_rank2 = agg_rank(df_f, [COL_PROD_ID, COL_PROD_DESC])

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Concentração: Top {top_n} vendedores", f"{share_top_n(vend_rank2, top_n):.2f}%")
    c2.metric(f"Concentração: Top {top_n} clientes", f"{share_top_n(cli_rank2, top_n):.2f}%")
    c3.metric(f"Concentração: Top {top_n} produtos", f"{share_top_n(prod_rank2, top_n):.2f}%")

    st.divider()

    st.subheader("🔥 Heatmap: Faturamento por Mês x Dia da Semana")
    h = heatmap_month_dow(df_f)
    if h.empty:
        st.info("Sem dados suficientes para heatmap.")
    else:
        pivot = h.pivot_table(index="mes_nome", columns="dia_semana", values="faturamento", aggfunc="sum").fillna(0.0)

        month_order = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        pivot = pivot.reindex([m for m in month_order if m in pivot.index])

        fig_hm = px.imshow(
            pivot,
            aspect="auto",
            labels={"x": "Dia da semana", "y": "Mês", "color": "Faturamento (R$)"},
            title="Onde o faturamento concentra (mês x dia da semana)"
        )
        fig_hm.update_layout(separators=",.")
        fig_hm.update_coloraxes(colorbar_tickprefix="R$ ", colorbar_tickformat=",.2f")
        st.plotly_chart(fig_hm, use_container_width=True, key=f"hm_{pivot.shape[0]}_{pivot.shape[1]}")

    st.divider()

    st.subheader("📌 Observação")
    ts_m = agg_timeseries(df_f, "M")
    if not ts_m.empty and ts_m["ano_mes"].nunique() >= 2:
        if exclude_incomplete_mom:
            m_last, is_complete, _ = get_last_month_info(df_f)
            if (m_last is not None) and (not is_complete):
                ts_m = ts_m[ts_m["ano_mes"] != m_last]

        if len(ts_m) >= 2:
            last = ts_m.iloc[-1]
            prev = ts_m.iloc[-2]
            if prev["faturamento"] > 0:
                delta = (last["faturamento"] - prev["faturamento"]) / prev["faturamento"] * 100
                st.write(
                    f"- Último mês considerado ({last['ano_mes']}) vs anterior ({prev['ano_mes']}): "
                    f"Δ {delta:.2f}% no faturamento."
                )
            else:
                st.write("- Não foi possível calcular variação: mês anterior com faturamento 0.")
        else:
            st.write("- Após excluir mês incompleto, não restaram 2 meses completos para comparação.")
    else:
        st.write("- Para variação mensal, o recorte precisa ter pelo menos 2 meses.")
