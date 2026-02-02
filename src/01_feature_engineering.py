from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class SalesOutputsGenerator:
    """
    Gera tabelas para Power BI a partir de vendas_enriched.csv.

    Modelo recomendado no Power BI (estrela):
    - fato_vendas (linha a linha / item)
    - dim_calendario (1 linha por DIA) -> relacionamento por data_key
    - dim_mes (1 linha por MÊS) -> relacionamento por ano_mes_key (opcional e útil)
    - dim_vendedor, dim_produto, dim_cliente, dim_filial, dim_geo

    Padrões:
    - ValorLiquido / ValorBruto / Desconto em CENTAVOS -> converte para REAIS
    - CSV padrão BR: sep=';', decimal=',', encoding='utf-8-sig'
    """

    root: Path
    processed_rel: Path = Path("data/processed")
    output_rel: Path = Path("data/output")
    input_filename: str = "vendas_enriched.csv"

    # colunas do CSV
    col_order_id: str = "Codigo"
    col_date: str = "DataVenda"

    col_filial: str = "CodFilial"

    col_cliente_id: str = "CodCliente"
    col_cliente_nome: str = "NomeCliente"
    col_cliente_cidade: str = "CidadeCliente"
    col_cliente_estado: str = "EstadoCliente"

    col_prod_id: str = "CodProduto"
    col_prod_desc: str = "DescricaoProdutos"

    col_vend_id: str = "CodVendedor"
    col_vend_nome: str = "NomeVendedor"

    col_valor_liq: str = "ValorLiquido"
    col_valor_bruto: str = "ValorBruto"
    col_desconto: str = "Desconto"

    # dinheiro: centavos -> reais
    money_scale: float = 100.0

    def __post_init__(self) -> None:
        self.processed_dir = self.root / self.processed_rel
        self.output_dir = self.root / self.output_rel
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.input_path = self.processed_dir / self.input_filename
        if not self.input_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.input_path}")

    # ============================================================
    # helpers
    # ============================================================
    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    @staticmethod
    def _br_regions_from_uf(uf: str) -> str:
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

    # ============================================================
    # LOAD + NORMALIZAÇÃO
    # ============================================================
    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_path)

        required = [
            self.col_order_id,
            self.col_date,
            self.col_filial,
            self.col_cliente_id,
            self.col_cliente_nome,
            self.col_cliente_cidade,
            self.col_cliente_estado,
            self.col_prod_id,
            self.col_prod_desc,
            self.col_vend_id,
            self.col_vend_nome,
            self.col_valor_liq,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        # data
        df[self.col_date] = pd.to_datetime(df[self.col_date], errors="coerce")
        df = df[df[self.col_date].notna()].copy()

        # dinheiro (centavos -> reais)
        for c in [self.col_valor_liq, self.col_valor_bruto, self.col_desconto]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                df[c] = (df[c] / self.money_scale).astype(float)

        # strings limpas
        for c in [
            self.col_cliente_nome,
            self.col_cliente_cidade,
            self.col_cliente_estado,
            self.col_prod_desc,
            self.col_vend_nome,
        ]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # derivadas de data
        df["ano"] = df[self.col_date].dt.year.astype(int)
        df["mes"] = df[self.col_date].dt.month.astype(int)
        df["dia"] = df[self.col_date].dt.day.astype(int)

        # chaves (IMPORTANTES pro BI)
        df["data_key"] = df[self.col_date].dt.strftime("%Y%m%d").astype(int)         # YYYYMMDD
        df["ano_mes_key"] = (df["ano"] * 100 + df["mes"]).astype(int)                # YYYYMM
        df["ano_mes"] = df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2)

        # dia da semana (ordem)
        df["dow_num"] = df[self.col_date].dt.dayofweek.astype(int)  # Monday=0...Sunday=6
        dow_names = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
        df["dia_semana"] = df["dow_num"].map(dow_names)

        # região via UF
        df["regiao"] = df[self.col_cliente_estado].apply(self._br_regions_from_uf)

        return df

    # ============================================================
    # DIMENSÕES
    # ============================================================
    def build_dim_calendario(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1 linha por DIA.
        Relacionamento recomendado no BI:
        dim_calendario[data_key] (1) -> fato_vendas[data_key] (*)
        """
        cal = (
            df[[self.col_date, "data_key", "ano", "mes", "dia", "ano_mes", "ano_mes_key", "dia_semana", "dow_num"]]
            .drop_duplicates()
            .sort_values(self.col_date)
            .reset_index(drop=True)
        )

        month_name = {
            1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
        }
        cal["mes_nome"] = cal["mes"].map(month_name)

        return cal

    def build_dim_mes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1 linha por MÊS.
        Serve para:
        - slicer de mês
        - gráfico de tendência mensal bonito
        Relacionamento opcional (recomendado):
        dim_mes[ano_mes_key] (1) -> fato_vendas[ano_mes_key] (*)
        """
        dim = (
            df[["ano_mes_key", "ano_mes", "ano", "mes"]]
            .drop_duplicates()
            .sort_values("ano_mes_key")
            .reset_index(drop=True)
        )

        month_name = {
            1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
        }
        dim["mes_nome"] = dim["mes"].map(month_name)
        dim["ano_mes_label"] = dim["mes_nome"] + "/" + dim["ano"].astype(str)

        return dim

    def build_dim_vendedor(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[[self.col_vend_id, self.col_vend_nome]]
            .drop_duplicates()
            .sort_values(self.col_vend_nome)
            .reset_index(drop=True)
        )

    def build_dim_produto(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[[self.col_prod_id, self.col_prod_desc]]
            .drop_duplicates()
            .sort_values(self.col_prod_desc)
            .reset_index(drop=True)
        )

    def build_dim_cliente(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[[self.col_cliente_id, self.col_cliente_nome, self.col_cliente_cidade, self.col_cliente_estado, "regiao"]]
            .drop_duplicates()
            .sort_values(self.col_cliente_nome)
            .reset_index(drop=True)
        )

    def build_dim_filial(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[[self.col_filial]]
            .drop_duplicates()
            .sort_values(self.col_filial)
            .reset_index(drop=True)
        )

    def build_dim_geo(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[[self.col_cliente_estado, "regiao"]]
            .drop_duplicates()
            .sort_values(["regiao", self.col_cliente_estado])
            .reset_index(drop=True)
        )

    # ============================================================
    # FATO
    # ============================================================
    def build_fato_vendas(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            self.col_order_id,
            self.col_date,
            "data_key",
            "ano",
            "mes",
            "dia",
            "ano_mes",
            "ano_mes_key",
            "dia_semana",
            "dow_num",
            self.col_filial,
            self.col_cliente_id,
            self.col_prod_id,
            self.col_vend_id,
            self.col_valor_liq,
        ]

        if self.col_valor_bruto in df.columns:
            cols.append(self.col_valor_bruto)
        if self.col_desconto in df.columns:
            cols.append(self.col_desconto)

        return df[cols].copy()

    # ============================================================
    # AGREGADOS (opcionais)
    # ============================================================
    @staticmethod
    def _agg(df: pd.DataFrame, group_cols: list[str], value_col: str, order_id_col: str) -> pd.DataFrame:
        out = (
            df.groupby(group_cols, as_index=False)
            .agg(
                faturamento=(value_col, "sum"),
                pedidos=(order_id_col, "nunique"),
                itens=(order_id_col, "count"),  # proxy de itens (linhas)
            )
        )
        out["faturamento"] = out["faturamento"].round(2)
        return out

    # ============================================================
    # PIPELINE
    # ============================================================
    def generate_all(self) -> dict[str, Path]:
        df = self.load()
        outputs: dict[str, Path] = {}

        # -------- DIMs + FATO (principal)
        dim_cal = self.build_dim_calendario(df)
        p = self.output_dir / "dim_calendario.csv"
        self._save_csv(dim_cal, p); outputs["dim_calendario"] = p

        dim_mes = self.build_dim_mes(df)
        p = self.output_dir / "dim_mes.csv"
        self._save_csv(dim_mes, p); outputs["dim_mes"] = p

        dim_vend = self.build_dim_vendedor(df)
        p = self.output_dir / "dim_vendedor.csv"
        self._save_csv(dim_vend, p); outputs["dim_vendedor"] = p

        dim_prod = self.build_dim_produto(df)
        p = self.output_dir / "dim_produto.csv"
        self._save_csv(dim_prod, p); outputs["dim_produto"] = p

        dim_cli = self.build_dim_cliente(df)
        p = self.output_dir / "dim_cliente.csv"
        self._save_csv(dim_cli, p); outputs["dim_cliente"] = p

        dim_filial = self.build_dim_filial(df)
        p = self.output_dir / "dim_filial.csv"
        self._save_csv(dim_filial, p); outputs["dim_filial"] = p

        dim_geo = self.build_dim_geo(df)
        p = self.output_dir / "dim_geo.csv"
        self._save_csv(dim_geo, p); outputs["dim_geo"] = p

        fato = self.build_fato_vendas(df)
        p = self.output_dir / "fato_vendas.csv"
        self._save_csv(fato, p); outputs["fato_vendas"] = p

        # -------- AGREGADOS (opcionais)
        vend_mes = self._agg(
            df,
            ["ano_mes_key", "ano_mes", "ano", "mes", self.col_vend_id, self.col_vend_nome],
            self.col_valor_liq,
            self.col_order_id
        ).sort_values("faturamento", ascending=False)
        p = self.output_dir / "vendedor_mes.csv"
        self._save_csv(vend_mes, p); outputs["vendedor_mes"] = p

        prod_mes = self._agg(
            df,
            ["ano_mes_key", "ano_mes", "ano", "mes", self.col_prod_id, self.col_prod_desc],
            self.col_valor_liq,
            self.col_order_id
        ).sort_values("faturamento", ascending=False)
        p = self.output_dir / "produto_mes.csv"
        self._save_csv(prod_mes, p); outputs["produto_mes"] = p

        filial_mes = self._agg(
            df,
            ["ano_mes_key", "ano_mes", "ano", "mes", self.col_filial],
            self.col_valor_liq,
            self.col_order_id
        ).sort_values("faturamento", ascending=False)
        p = self.output_dir / "filial_mes.csv"
        self._save_csv(filial_mes, p); outputs["filial_mes"] = p

        cliente_mes = self._agg(
            df,
            ["ano_mes_key", "ano_mes", "ano", "mes", self.col_cliente_id, self.col_cliente_nome],
            self.col_valor_liq,
            self.col_order_id
        ).sort_values("faturamento", ascending=False)
        p = self.output_dir / "cliente_mes.csv"
        self._save_csv(cliente_mes, p); outputs["cliente_mes"] = p

        geo_mes = self._agg(
            df,
            ["ano_mes_key", "ano_mes", "ano", "mes", self.col_cliente_estado, "regiao"],
            self.col_valor_liq,
            self.col_order_id
        ).sort_values("faturamento", ascending=False)
        p = self.output_dir / "geo_mes.csv"
        self._save_csv(geo_mes, p); outputs["geo_mes"] = p

        dow = self._agg(
            df,
            ["dia_semana", "dow_num"],
            self.col_valor_liq,
            self.col_order_id
        ).sort_values("dow_num")
        p = self.output_dir / "dia_semana.csv"
        self._save_csv(dow, p); outputs["dia_semana"] = p

        return outputs


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    gen = SalesOutputsGenerator(root=ROOT)
    outputs = gen.generate_all()
    print("✅ Outputs gerados:")
    for k, v in outputs.items():
        print(f"- {k}: {v}")
