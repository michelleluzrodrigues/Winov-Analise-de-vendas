# Análise de Vendas — Python, Power BI e Streamlit

Projeto para **preparação de dados de vendas**, **dashboard interativo em Streamlit** e **Geração da iamgensdos reports**.

---

## 📁 Estrutura do Projeto

```
.
├─ data/
│  ├─ raw/                # Dados brutos (originais)
│  ├─ processed/          # Dados processados (ex: vendas_enriched.csv)
│  ├─ output/             
│  └─ reports/            # Relatórios e assets gerados
├─ notebook/
│  └─ 01_eda_exploratoria.ipynb
├─ reports/
│  └─ Relatório de Vendas.pdf
├─ src/
│  ├─ 01_feature_engineering.py
│  ├─ 02_dashboard_streamlit.py
│  └─ 03_export_report.py
├─ requirements.txt
└─ README.md
```

---

## ✅ Pré-requisitos

- Python **3.10+**
- Pip

---

## 🧪 Criando e ativando o ambiente virtual (venv)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Se houver bloqueio de execução:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Windows (CMD)

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dados de Entrada

Arquivo principal esperado:

```
data/processed/vendas_enriched.csv
```

---

## ▶️ Como rodar o projeto

### 1️⃣ Feature Engineering 

Criar novas “features” (colunas/variáveis) em cima dos dados originais, para deixar o dataset mais útil para análise.

```bash
python src/01_feature_engineering.py
```

**Saída:**
data/processed/vendas_enriched.csv
---

### 2️⃣ Dashboard Interativo (Streamlit)

Dashboard com filtros, KPIs, rankings e análises temporais.

```bash
streamlit run src/02_dashboard_streamlit.py
```

Depois, acesse no navegador:
```
http://localhost:8501
```

---

### 3️⃣ Exportação de Relatório

Gera gráficos (.png).

```bash
python src/03_export_report.py
```

Com parâmetros opcionais:

```bash
python src/03_export_report.py   --input data/processed/vendas_enriched.csv   --outdir data/reports   --money-in-cents
```

---

##  Observações

- Se os valores monetários estiverem em **centavos**, utilize `--money-in-cents`
- Sempre execute os scripts com o **venv ativado**

---

