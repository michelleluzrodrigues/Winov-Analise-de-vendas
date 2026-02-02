from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# =========================
# CONFIG PADRÃO (colunas)
# =========================
COL_DATE = "DataVenda"
COL_VEND = "NomeVendedor"
COL_PROD = "DescricaoProdutos"
COL_CLIENTE = "NomeCliente"
COL_UF = "EstadoCliente"
COL_CIDADE = "CidadeCliente"
COL_FILIAL = "CodFilial"
COL_PEDIDO = "Codigo"
COL_VALOR = "ValorLiquido"
COL_BRUTO = "ValorBruto"
COL_DESC = "Desconto"
COL_VARIACAO = "Variação"
COL_ANO = "ano"
COL_MES = "mes"
COL_ANO_MES = "ano_mes"
COL_DESC_PCT = "desconto_produto"

WEEKDAY_NAMES = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}

DEFAULT_INPUT = Path(__file__).resolve().parent / "../data/processed/vendas_enriched.csv"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "../data/reports"


def brl_formatter():
    def _fmt(v, _pos):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        s = f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"
    return FuncFormatter(_fmt)


@dataclass
class SalesReportGenerator:
    input_path: Path
    outdir: Path
    money_in_cents: bool = False  # se True: divide ValorLiquido/Bruto/Desconto por 100

    def run(self) -> None:
        self.outdir = self.outdir.resolve()
        self.input_path = self.input_path.resolve()

        if not self.input_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.input_path}")

        self._ensure_outdir()

        df = self._read_csv(self.input_path)
        df = self._normalize(df)

        self._make_views(df)

        issues = self._issues_report(df)
        (self.outdir / "issues.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.outdir / "issues.txt").write_text(
            "\n".join([f"- {i['issue']} | count={i['count']} | details={str(i['details'])[:240]}..." for i in issues]),
            encoding="utf-8",
        )

        summary = self._summary(df)
        (self.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n✅ Concluído!")
        print(f"- Input:   {self.input_path}")
        print(f"- Outdir:  {self.outdir}")
        print(f"- Imagens: {self.outdir / 'report_assets'}")
        print(f"- Tabelas: {self.outdir / 'report_tables'}")
        print(f"- Issues:  {self.outdir / 'issues.txt'} e {self.outdir / 'issues.json'}")
        print(f"- Summary: {self.outdir / 'summary.json'}\n")

    # -------------------------
    # IO / Setup
    # -------------------------
    def _ensure_outdir(self) -> None:
        (self.outdir / "report_assets").mkdir(parents=True, exist_ok=True)
        (self.outdir / "report_tables").mkdir(parents=True, exist_ok=True)

    def _read_csv(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
            if df.shape[1] > 3:
                return df
        except Exception:
            pass
        return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")

    # -------------------------
    # Normalização
    # -------------------------
    def _clean_text_col(self, df: pd.DataFrame, col: str) -> None:
        if col not in df.columns:
            return
        df[col] = (
            df[col]
            .astype(str)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            .str.strip()
        )

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # limpa colunas de texto (evita "nan" como string e espaços)
        for c in [COL_VEND, COL_PROD, COL_CLIENTE, COL_UF, COL_CIDADE]:
            self._clean_text_col(df, c)

        if COL_DATE in df.columns:
            df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

        if COL_DATE in df.columns:
            if COL_ANO not in df.columns:
                df[COL_ANO] = df[COL_DATE].dt.year
            if COL_MES not in df.columns:
                df[COL_MES] = df[COL_DATE].dt.month
            if COL_ANO_MES not in df.columns:
                df[COL_ANO_MES] = df[COL_DATE].dt.to_period("M").astype(str)

            if "dia_semana" not in df.columns:
                df["dia_semana"] = df[COL_DATE].dt.weekday
            if "dia_semana_nome" not in df.columns:
                df["dia_semana_nome"] = df["dia_semana"].map(WEEKDAY_NAMES)

        for c in [COL_VALOR, COL_BRUTO, COL_DESC, COL_VARIACAO, COL_DESC_PCT]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if self.money_in_cents:
            for c in [COL_VALOR, COL_BRUTO, COL_DESC]:
                if c in df.columns:
                    df[c] = df[c] / 100.0

        return df

    # -------------------------
    # Helpers de plot
    # -------------------------
    def _period_label(self, df: pd.DataFrame) -> str:
        if COL_DATE not in df.columns or df[COL_DATE].isna().all():
            return ""
        dmin = df[COL_DATE].min()
        dmax = df[COL_DATE].max()
        if pd.isna(dmin) or pd.isna(dmax):
            return ""
        return f"{dmin:%d/%m/%Y}–{dmax:%d/%m/%Y}"

    def _save_fig(
        self,
        path: Path,
        title: str,
        money_axis: str | None = None,  # ✅ "x" ou "y" ou None
        xlabel: str | None = None,
        ylabel: str | None = None,
    ):
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        ax = plt.gca()
        ax.grid(axis="y", alpha=0.2)

        if money_axis == "y":
            ax.yaxis.set_major_formatter(brl_formatter())
        elif money_axis == "x":
            ax.xaxis.set_major_formatter(brl_formatter())

        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    # -------------------------
    # Views
    # -------------------------
    def _make_views(self, df: pd.DataFrame) -> None:
        assets = self.outdir / "report_assets"
        tables = self.outdir / "report_tables"

        base = df.copy()
        if COL_DATE in base.columns:
            base = base[base[COL_DATE].notna()]
        if COL_VALOR in base.columns:
            base = base[base[COL_VALOR].notna()]

        period = self._period_label(base)
        period_suffix = f" ({period})" if period else ""

        # VIEW 01: Receita por mês (linha) -> dinheiro no Y
        if COL_ANO_MES in base.columns and COL_VALOR in base.columns:
            s = base.groupby(COL_ANO_MES)[COL_VALOR].sum().sort_index()
            plt.figure(figsize=(10, 4))
            s.plot()
            self._save_fig(
                assets / "01_receita_por_mes.png",
                f"Receita por mês{period_suffix}",
                money_axis="y",
                xlabel="Ano-Mês",
                ylabel="Receita (R$)",
            )
            s.reset_index().rename(columns={COL_VALOR: "Receita"}).to_csv(tables / "receita_por_mes.csv", index=False)

        # VIEW 02: Top 10 vendedores (barh) -> dinheiro no X
        if COL_VEND in base.columns and COL_VALOR in base.columns:
            top = (
                base.dropna(subset=[COL_VEND])
                .groupby(COL_VEND)[COL_VALOR].sum()
                .sort_values(ascending=False)
                .head(10)
            )
            plt.figure(figsize=(10, 5))
            top.sort_values().plot(kind="barh")
            self._save_fig(
                assets / "02_top10_vendedores_geral.png",
                f"Top 10 vendedores (Receita total){period_suffix}",
                money_axis="x",
                xlabel="Receita (R$)",
                ylabel="Vendedor",
            )
            top.reset_index().rename(columns={COL_VALOR: "Receita"}).to_csv(tables / "top10_vendedores_geral.csv", index=False)

        # VIEW 03: Top 5 vendedores por ano (barh) -> dinheiro no X
        if COL_VEND in base.columns and COL_ANO in base.columns and COL_VALOR in base.columns:
            by = (
                base.dropna(subset=[COL_VEND, COL_ANO])
                .groupby([COL_ANO, COL_VEND])[COL_VALOR].sum()
                .reset_index()
                .sort_values([COL_ANO, COL_VALOR], ascending=[True, False])
            )
            by.to_csv(tables / "vendedor_ano_receita.csv", index=False)

            top5 = by.groupby(COL_ANO).head(5)
            for ano, g in top5.groupby(COL_ANO):
                g2 = g.set_index(COL_VEND)[COL_VALOR].sort_values()
                plt.figure(figsize=(10, 4))
                g2.plot(kind="barh")
                self._save_fig(
                    assets / f"03_top5_vendedores_{int(ano)}.png",
                    f"Top 5 vendedores em {int(ano)}",
                    money_axis="x",
                    xlabel="Receita (R$)",
                    ylabel="Vendedor",
                )

        # VIEW 04: Receita por UF (barh) -> dinheiro no X
        if COL_UF in base.columns and COL_VALOR in base.columns:
            uf = (
                base.dropna(subset=[COL_UF])
                .groupby(COL_UF)[COL_VALOR].sum()
                .sort_values(ascending=False)
                .head(10)
            )
            plt.figure(figsize=(10, 5))
            uf.sort_values().plot(kind="barh")
            self._save_fig(
                assets / "04_receita_por_uf_top10.png",
                f"Receita por UF (Top 10){period_suffix}",
                money_axis="x",
                xlabel="Receita (R$)",
                ylabel="UF",
            )
            uf.reset_index().rename(columns={COL_VALOR: "Receita"}).to_csv(tables / "receita_por_uf_top10.csv", index=False)

        # VIEW 05: Distribuição de desconto (%)
        if COL_DESC_PCT in base.columns:
            x = base[COL_DESC_PCT].dropna()
            if len(x) > 0:
                plt.figure(figsize=(10, 4))
                plt.hist(x, bins=25)
                self._save_fig(
                    assets / "05_distribuicao_desconto_pct.png",
                    f"Distribuição do desconto (%) {period_suffix}".strip(),
                    money_axis=None,
                    xlabel="Desconto (%)",
                    ylabel="Frequência",
                )
                x.describe().to_frame("stats").to_csv(tables / "stats_desconto_pct.csv")

        # VIEW 06: Top 10 produtos (barh) -> dinheiro no X
        if COL_PROD in base.columns and COL_VALOR in base.columns:
            prod = (
                base.dropna(subset=[COL_PROD])
                .groupby(COL_PROD)[COL_VALOR].sum()
                .sort_values(ascending=False)
                .head(10)
            )
            plt.figure(figsize=(10, 5))
            prod.sort_values().plot(kind="barh")
            self._save_fig(
                assets / "06_top10_produtos.png",
                f"Top 10 produtos (Receita){period_suffix}",
                money_axis="x",
                xlabel="Receita (R$)",
                ylabel="Produto",
            )
            prod.reset_index().rename(columns={COL_VALOR: "Receita"}).to_csv(tables / "top10_produtos.csv", index=False)

        # VIEW 07: Top 10 clientes (barh) -> dinheiro no X
        if COL_CLIENTE in base.columns and COL_VALOR in base.columns:
            cli = (
                base.dropna(subset=[COL_CLIENTE])
                .groupby(COL_CLIENTE)[COL_VALOR].sum()
                .sort_values(ascending=False)
                .head(10)
            )
            plt.figure(figsize=(10, 5))
            cli.sort_values().plot(kind="barh")
            self._save_fig(
                assets / "07_top10_clientes.png",
                f"Top 10 clientes (Receita){period_suffix}",
                money_axis="x",
                xlabel="Receita (R$)",
                ylabel="Cliente",
            )
            cli.reset_index().rename(columns={COL_VALOR: "Receita"}).to_csv(tables / "top10_clientes.csv", index=False)

        # VIEW 08: Pedidos por dia da semana (bar) -> valores no Y (qtd)
        if "dia_semana" in base.columns and COL_PEDIDO in base.columns:
            wk = base.groupby("dia_semana")[COL_PEDIDO].nunique().reindex(range(7))
            wk.index = [WEEKDAY_NAMES.get(i, str(i)) for i in wk.index]
            plt.figure(figsize=(10, 4))
            wk.plot(kind="bar")
            self._save_fig(
                assets / "08_pedidos_por_dia_semana.png",
                f"Quantidade de pedidos por dia da semana{period_suffix}",
                money_axis=None,
                xlabel="Dia da semana",
                ylabel="Pedidos (qtd.)",
            )
            wk.reset_index().rename(columns={"index": "dia_semana", COL_PEDIDO: "Pedidos"}).to_csv(
                tables / "pedidos_por_dia_semana.csv", index=False
            )

    # -------------------------
    # Issues / Summary (mantive como estava)
    # -------------------------
    def _issues_report(self, df: pd.DataFrame) -> list[dict]:
        issues: list[dict] = []

        def add_issue(name: str, count: int, details: dict):
            if count > 0:
                issues.append({"issue": name, "count": int(count), "details": details})

        if COL_DATE in df.columns:
            n_bad = int(df[COL_DATE].isna().sum())
            add_issue(
                "datas_invalidas_em_DataVenda",
                n_bad,
                {"exemplos_linhas": df[df[COL_DATE].isna()].head(5).to_dict(orient="records")},
            )

        if COL_VALOR in df.columns:
            add_issue("valorliquido_negativo", int((df[COL_VALOR] < 0).sum()), {"coluna": COL_VALOR})
            add_issue("valorliquido_zero", int((df[COL_VALOR] == 0).sum()), {"coluna": COL_VALOR})

        if COL_DESC in df.columns:
            add_issue("desconto_negativo", int((df[COL_DESC] < 0).sum()), {"coluna": COL_DESC})

        if COL_VALOR in df.columns and COL_BRUTO in df.columns:
            mask = df[COL_VALOR] > df[COL_BRUTO]
            add_issue(
                "valorliquido_maior_que_valorbruto",
                int(mask.sum()),
                {"exemplos": df[mask].head(5)[[COL_PEDIDO, COL_BRUTO, COL_VALOR, COL_DESC]].to_dict(orient="records")},
            )

        if COL_UF in df.columns:
            uf = df[COL_UF].astype(str)
            mask = (uf.str.lower().isin(["nan", "none", ""])) | (uf.str.len() != 2)
            add_issue(
                "estado_cliente_invalido",
                int(mask.sum()),
                {"exemplos": df[mask].head(10)[[COL_PEDIDO, COL_UF, COL_CIDADE, COL_CLIENTE]].to_dict(orient="records")},
            )

        if COL_PEDIDO in df.columns:
            dups = df[COL_PEDIDO].duplicated(keep=False)
            add_issue(
                "codigo_pedido_duplicado",
                int(dups.sum()),
                {"exemplos": df[dups].head(10)[[COL_PEDIDO, COL_DATE, COL_VALOR, COL_VEND]].to_dict(orient="records")},
            )

        critical_cols = [COL_DATE, COL_VEND, COL_CLIENTE, COL_PROD, COL_VALOR]
        present = [c for c in critical_cols if c in df.columns]
        if present:
            miss = df[present].isna().sum().sort_values(ascending=False)
            for c, n in miss.items():
                add_issue(f"missing_em_{c}", int(n), {"coluna": c})

        return issues

    def _summary(self, df: pd.DataFrame) -> dict:
        return {
            "linhas": int(df.shape[0]),
            "colunas": int(df.shape[1]),
            "periodo_min": str(df[COL_DATE].min()) if COL_DATE in df.columns else None,
            "periodo_max": str(df[COL_DATE].max()) if COL_DATE in df.columns else None,
            "receita_total": float(df[COL_VALOR].sum()) if COL_VALOR in df.columns else None,
            "qtd_pedidos": int(df[COL_PEDIDO].nunique()) if COL_PEDIDO in df.columns else None,
            "qtd_vendedores": int(df[COL_VEND].nunique()) if COL_VEND in df.columns else None,
            "qtd_clientes": int(df[COL_CLIENTE].nunique()) if COL_CLIENTE in df.columns else None,
            "qtd_produtos": int(df[COL_PROD].nunique()) if COL_PROD in df.columns else None,
            "money_in_cents": bool(self.money_in_cents),
        }


def main():
    parser = argparse.ArgumentParser(description="Gera imagens (dashboards) + tabelas + inconsistências a partir do CSV.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--money-in-cents", action="store_true")
    args = parser.parse_args()

    SalesReportGenerator(
        input_path=Path(args.input),
        outdir=Path(args.outdir),
        money_in_cents=args.money_in_cents,
    ).run()


if __name__ == "__main__":
    main()
