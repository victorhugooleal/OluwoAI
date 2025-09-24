---
applyTo: '**'
---


# INSTRUÇÕES POC BARATA - OLUWO AI

---

## 1-visao-geral
POC funcional, zero custo, sem dependências pagas. Demonstra busca semântica e respostas contextualizadas sobre os Odus usando recursos 100% gratuitos/open-source.

## 2-stack-tecnico-sugerido
- **Linguagem:** Python 3.10+
- **Interface:** Streamlit (rápida, local)
- **LLM:** Ollama local (Llama 3.2 3B, Mistral 7B, Phi3 Mini)
- **Embeddings:** sentence-transformers (MiniLM, local)
- **Banco Vetorial:** ChromaDB ou FAISS (local, em memória)
- **Processamento PDF:** PyPDF2 + pdfplumber
- **Cache:** dict em memória ou SQLite
- **Deploy:** Local (Windows/Linux/Mac)

## 3-pipeline-simplificado
1. Processar PDFs dos Odus para JSON estruturado usando PyPDF2/pdfplumber
2. Gerar chunks e embeddings com sentence-transformers
3. Indexar chunks dos Odus no ChromaDB/FAISS local
4. Busca semântica simples (top-k)
5. Gerar resposta usando LLM local via Ollama
6. Citar Odu e verso na resposta

## 4-estrutura-de-pastas-sugerida
```
oluwo-ia-poc/
├── app.py                 # Interface Streamlit
├── processor.py           # Processamento de PDFs
├── indexer.py             # Sistema de busca
├── llm_handler.py         # Gerenciador do Ollama
├── data/
│   ├── odus_pdf/          # 256 PDFs originais
│   ├── processed/         # JSONs processados
│   └── vector_db/         # Base vetorial
├── prompts/
│   └── templates.py       # Templates de prompts
└── utils/
    └── helpers.py         # Funções auxiliares
```

## 5-requisitos-e-instalacao

### requirements.txt
```txt
streamlit==1.28.0
ollama==0.1.8
chromadb==0.4.18
sentence-transformers==2.2.2
pdfplumber==0.10.3
PyPDF2==3.0.1
pandas==2.1.3
numpy==1.24.3
python-dotenv==1.0.0
langchain==0.1.0
langchain-community==0.0.10
```

### Instalação rápida
```bash
# 1. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# 2. Instale dependências
pip install -r requirements.txt

# 3. Instale Ollama manualmente
# Mac/Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: baixe de https://ollama.com/download
```

**Links úteis:**
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://docs.trychroma.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)

> **Atenção:** Os arquivos PDF dos Odus não devem ser versionados no GitHub. Mantenha-os apenas na pasta local `data/odus_pdf/`. O .gitignore já está configurado para isso.

## 6-implementacao-principal

### Exemplos completos de código
Os exemplos completos de cada módulo estão descritos no texto do Claude Opus abaixo deste instructions. Recomenda-se criar um diretório `examples/` ou consultar os arquivos `processor.py`, `indexer.py`, `llm_handler.py` e `app.py` conforme a estrutura sugerida.

#### Processamento dos PDFs (`processor.py`)
```python
# ...veja exemplo completo no texto do Claude abaixo...
```

#### Indexação local (`indexer.py`)
```python
# ...veja exemplo completo no texto do Claude abaixo...
```

#### Gerenciador LLM local (`llm_handler.py`)
```python
# ...veja exemplo completo no texto do Claude abaixo...
```

#### Interface Streamlit (`app.py`)
```python
# ...veja exemplo completo no texto do Claude abaixo...
```


## 7-uso-da-interface
1. Coloque os PDFs dos Odus em `data/odus_pdf/`
2. Execute:
    ```bash
    streamlit run app.py
    ```
3. Use a interface para processar, indexar e consultar os Odus


## 8-otimizacoes-para-poc
- Use modelo menor se tiver pouca RAM: `phi3:mini`
- Reduza chunk_size se necessário
- Use FAISS se precisar de menos memória
- Cache simples com `@lru_cache`


## 9-custos-e-requisitos-de-hardware
- LLM, embeddings, vector DB, interface: **Gratuito**
- RAM mínima: 8GB (recomendado 16GB)
- Disco: 10GB (recomendado 20GB)
- GPU: opcional


## 10-proximos-passos-apos-poc
- Deploy gratuito: Hugging Face Spaces, Railway.app
- Versão produção: API robusta, LLM mais poderosa, deploy em cloud


## 11-consideracoes-finais
- POC pode ser expandida para versão paga no futuro
- Foco em simplicidade, baixo custo e funcionalidade básica
- Interpretações devem ser validadas por babalaôs experientes

---


texto abaixo foi gerado pelo claude opus quero que leia ele e incremente na estrutura acima:

POC Oluwo IA - Versão Zero Cost
🎯 ESTRATÉGIA PARA POC GRATUITA
Vou reformular completamente o projeto para uma POC 100% gratuita que você pode rodar localmente e demonstrar o conceito.
1. ARQUITETURA SIMPLIFICADA (100% LOCAL E GRATUITA)
yaml# Stack para POC sem custos
linguagem: Python 3.10+
llm: Ollama com Llama 3.2 3B ou Mistral 7B (local)
embeddings: sentence-transformers (local)
vector_db: ChromaDB ou FAISS (local, em memória)
processamento_pdf: PyPDF2 + pdfplumber
web_framework: Streamlit (interface rápida)
cache: Simples dict em memória
deployment: Local apenas
2. SETUP INICIAL MINIMALISTA
2.1 Requirements.txt
txtstreamlit==1.28.0
ollama==0.1.8
chromadb==0.4.18
sentence-transformers==2.2.2
pdfplumber==0.10.3
PyPDF2==3.0.1
pandas==2.1.3
numpy==1.24.3
python-dotenv==1.0.0
langchain==0.1.0
langchain-community==0.0.10
2.2 Estrutura de Pastas
oluwo-ia-poc/
├── app.py                 # Interface Streamlit
├── processor.py           # Processamento de PDFs
├── indexer.py            # Sistema de busca
├── llm_handler.py        # Gerenciador do Ollama
├── data/
│   ├── odus_pdf/         # 256 PDFs originais
│   ├── processed/        # JSONs processados
│   └── vector_db/        # Base vetorial
├── prompts/
│   └── templates.py      # Templates de prompts
└── utils/
    └── helpers.py        # Funções auxiliares
3. IMPLEMENTAÇÃO DA POC
3.1 Processador de PDFs (processor.py)
pythonimport pdfplumber
import json
import re
from pathlib import Path
from typing import Dict, List
import hashlib

class OduPDFProcessor:
    def __init__(self, input_dir: str = "data/odus_pdf"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """Processa um único PDF de Odu"""
        print(f"📖 Processando: {pdf_path.name}")
        
        odu_data = {
            "file_name": pdf_path.name,
            "odu_id": self.generate_id(pdf_path.name),
            "full_text": "",
            "chunks": []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        full_text += f"\n--- Página {page_num} ---\n{text}"
                
                odu_data["full_text"] = full_text
                
                # Criar chunks menores para indexação
                odu_data["chunks"] = self.create_chunks(full_text, pdf_path.name)
                
                # Extrair metadados básicos
                odu_data["metadata"] = self.extract_metadata(full_text)
                
        except Exception as e:
            print(f"❌ Erro ao processar {pdf_path.name}: {e}")
            
        return odu_data
    
    def create_chunks(self, text: str, filename: str, chunk_size: int = 500) -> List[Dict]:
        """Divide o texto em chunks menores para melhor indexação"""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "chunk_id": f"{filename}_{chunk_id}",
                        "text": current_chunk.strip(),
                        "metadata": {
                            "source": filename,
                            "chunk_number": chunk_id
                        }
                    })
                    chunk_id += 1
                current_chunk = para + "\n\n"
        
        # Adicionar último chunk
        if current_chunk:
            chunks.append({
                "chunk_id": f"{filename}_{chunk_id}",
                "text": current_chunk.strip(),
                "metadata": {
                    "source": filename,
                    "chunk_number": chunk_id
                }
            })
            
        return chunks
    
    def extract_metadata(self, text: str) -> Dict:
        """Extrai metadados básicos do texto"""
        metadata = {
            "orisas": [],
            "ebos": [],
            "keywords": []
        }
        
        # Patterns simplificados para POC
        orisa_pattern = r'(Ifá|Orí|Èṣù|Ọbàtálá|Ògún|Ọ̀ṣun|Yemọja|Ṣàngó)'
        ebo_pattern = r'ẹbọ.*?(?:com|para).*?(?:\.|;|\n)'
        
        metadata["orisas"] = list(set(re.findall(orisa_pattern, text, re.IGNORECASE)))
        metadata["ebos"] = re.findall(ebo_pattern, text, re.IGNORECASE)[:5]
        
        return metadata
    
    def generate_id(self, filename: str) -> str:
        """Gera ID único para o Odu"""
        return hashlib.md5(filename.encode()).hexdigest()[:12]
    
    def process_all_pdfs(self):
        """Processa todos os PDFs na pasta"""
        all_odus = []
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        print(f"🔍 Encontrados {len(pdf_files)} PDFs para processar")
        
        for pdf_path in pdf_files:
            odu_data = self.process_single_pdf(pdf_path)
            all_odus.append(odu_data)
            
            # Salvar individual
            output_file = self.output_dir / f"{odu_data['odu_id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(odu_data, f, ensure_ascii=False, indent=2)
        
        # Salvar compilado
        with open(self.output_dir / "all_odus.json", 'w', encoding='utf-8') as f:
            json.dump(all_odus, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Processamento concluído! {len(all_odus)} Odus processados")
        return all_odus
3.2 Sistema de Indexação Local (indexer.py)
pythonimport chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from typing import List, Dict

class OduLocalIndexer:
    def __init__(self, persist_directory: str = "data/vector_db"):
        # Configurar ChromaDB local
        self.persist_dir = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Modelo de embeddings multilíngue pequeno e eficiente
        print("🔄 Carregando modelo de embeddings...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Criar ou carregar coleção
        try:
            self.collection = self.client.create_collection(
                name="odus_collection",
                metadata={"hnsw:space": "cosine"}
            )
            print("📚 Nova coleção criada")
        except:
            self.collection = self.client.get_collection("odus_collection")
            print("📚 Coleção existente carregada")
    
    def index_odus(self, json_dir: str = "data/processed"):
        """Indexa todos os Odus processados"""
        json_path = Path(json_dir) / "all_odus.json"
        
        if not json_path.exists():
            print("❌ Arquivo all_odus.json não encontrado. Execute o processamento primeiro.")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            odus = json.load(f)
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        print(f"🔄 Indexando {len(odus)} Odus...")
        
        for odu in odus:
            for chunk in odu['chunks']:
                # Preparar documento
                doc_text = chunk['text']
                documents.append(doc_text)
                
                # Gerar embedding
                embedding = self.embedder.encode(doc_text).tolist()
                embeddings.append(embedding)
                
                # Metadados
                metadata = {
                    "source": chunk['metadata']['source'],
                    "chunk_id": chunk['chunk_id'],
                    "odu_id": odu['odu_id']
                }
                metadatas.append(metadata)
                
                # ID único
                ids.append(chunk['chunk_id'])
        
        # Adicionar à coleção em batch
        print(f"💾 Salvando {len(documents)} chunks no banco vetorial...")
        
        # ChromaDB tem limite de batch, então dividimos
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            print(f"  Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} salvo")
        
        print(f"✅ Indexação concluída! {len(documents)} chunks indexados")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Busca semântica nos Odus"""
        # Gerar embedding da query
        query_embedding = self.embedder.encode(query).tolist()
        
        # Buscar no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Formatar resultados
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted_results
3.3 Gerenciador LLM Local (llm_handler.py)
pythonimport ollama
import json
from typing import Dict, List

class OllamaHandler:
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Modelos recomendados para POC (escolha um):
        - llama3.2:3b (3GB, mais rápido)
        - mistral:7b (4GB, mais capaz)
        - phi3:mini (2GB, super leve)
        """
        self.model_name = model_name
        self.ensure_model_downloaded()
        
    def ensure_model_downloaded(self):
        """Verifica e baixa o modelo se necessário"""
        try:
            # Verificar se modelo existe
            ollama.show(self.model_name)
            print(f"✅ Modelo {self.model_name} já está disponível")
        except:
            print(f"📥 Baixando modelo {self.model_name}... (pode demorar na primeira vez)")
            ollama.pull(self.model_name)
            print(f"✅ Modelo {self.model_name} baixado com sucesso")
    
    def generate_response(self, 
                         query: str, 
                         contexts: List[Dict],
                         temperature: float = 0.7) -> Dict:
        """Gera resposta baseada nos contextos encontrados"""
        
        # Preparar contexto
        context_text = self._prepare_context(contexts)
        
        # Prompt otimizado para modelo pequeno
        prompt = f"""Você é Oluwo IA, um intérprete dos Odus de Ifá.

CONTEXTOS DOS ODUS:
{context_text}

PERGUNTA DO USUÁRIO:
{query}

INSTRUÇÕES:
1. Responda baseando-se APENAS nos contextos fornecidos
2. Cite o nome do arquivo/Odu quando relevante
3. Se não encontrar informação relevante, diga que não encontrou
4. Seja conciso mas completo

RESPOSTA:"""

        try:
            # Gerar resposta com Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'max_tokens': 500,
                    'top_p': 0.9
                }
            )
            
            return {
                'answer': response['response'],
                'model': self.model_name,
                'contexts_used': len(contexts)
            }
            
        except Exception as e:
            return {
                'answer': f"Erro ao gerar resposta: {str(e)}",
                'model': self.model_name,
                'contexts_used': 0
            }
    
    def _prepare_context(self, contexts: List[Dict], max_length: int = 2000) -> str:
        """Prepara e limita o contexto para o modelo"""
        context_parts = []
        current_length = 0
        
        for i, ctx in enumerate(contexts, 1):
            text = ctx['text']
            source = ctx['metadata'].get('source', 'Desconhecido')
            
            # Limitar tamanho do contexto
            if current_length + len(text) > max_length:
                text = text[:max_length - current_length]
                
            context_parts.append(f"[Fonte: {source}]\n{text}\n")
            current_length += len(text)
            
            if current_length >= max_length:
                break
        
        return "\n---\n".join(context_parts)
3.4 Interface Streamlit (app.py)
pythonimport streamlit as st
from pathlib import Path
import time
from processor import OduPDFProcessor
from indexer import OduLocalIndexer
from llm_handler import OllamaHandler

# Configuração da página
st.set_page_config(
    page_title="Oluwo IA - POC",
    page_icon="🔮",
    layout="wide"
)

# Inicializar session state
if 'indexer' not in st.session_state:
    st.session_state.indexer = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Header
st.title("🔮 Oluwo IA - Intérprete dos 256 Odus")
st.markdown("*Sabedoria ancestral Yorubá através de IA*")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Seletor de modelo
    model_option = st.selectbox(
        "Modelo LLM Local",
        ["llama3.2:3b", "mistral:7b", "phi3:mini"],
        help="Escolha o modelo. Modelos menores são mais rápidos."
    )
    
    # Número de resultados
    n_results = st.slider(
        "Número de contextos",
        min_value=1,
        max_value=10,
        value=5,
        help="Quantos trechos buscar nos Odus"
    )
    
    st.divider()
    
    # Seção de Setup
    st.header("🚀 Setup Inicial")
    
    if st.button("1️⃣ Processar PDFs", type="primary"):
        with st.spinner("Processando PDFs..."):
            processor = OduPDFProcessor()
            
            # Verificar se há PDFs
            pdf_dir = Path("data/odus_pdf")
            if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
                st.error("⚠️ Coloque os PDFs dos Odus na pasta `data/odus_pdf/`")
            else:
                odus = processor.process_all_pdfs()
                st.success(f"✅ {len(odus)} Odus processados!")
                st.session_state.processed = True
    
    if st.button("2️⃣ Criar Índice Vetorial"):
        if not st.session_state.processed:
            # Verificar se já foi processado anteriormente
            if not Path("data/processed/all_odus.json").exists():
                st.error("❌ Processe os PDFs primeiro!")
            else:
                st.session_state.processed = True
        
        if st.session_state.processed:
            with st.spinner("Criando índice vetorial..."):
                st.session_state.indexer = OduLocalIndexer()
                st.session_state.indexer.index_odus()
                st.success("✅ Índice criado!")
    
    if st.button("3️⃣ Inicializar LLM"):
        with st.spinner(f"Inicializando {model_option}..."):
            st.session_state.llm = OllamaHandler(model_option)
            st.success(f"✅ {model_option} pronto!")
    
    # Status do sistema
    st.divider()
    st.header("📊 Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.processed:
            st.success("✅ PDFs")
        else:
            st.info("⏳ PDFs")
    
    with col2:
        if st.session_state.indexer:
            st.success("✅ Índice")
        else:
            st.info("⏳ Índice")
    
    if st.session_state.llm:
        st.success(f"✅ LLM: {model_option}")
    else:
        st.info("⏳ LLM")

# Área principal
if not st.session_state.indexer or not st.session_state.llm:
    st.info("👈 Configure o sistema na barra lateral antes de fazer consultas")
else:
    # Interface de consulta
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Consultar os Odus")
        
        # Input da pergunta
        query = st.text_area(
            "Faça sua pergunta aos Odus:",
            placeholder="Ex: O que os Odus dizem sobre prosperidade? Como lidar com dívidas? Qual orientação para novos começos?",
            height=100
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
        with col_btn1:
            search_button = st.button("🔮 Consultar", type="primary")
        with col_btn2:
            clear_button = st.button("🧹 Limpar")
        
        if clear_button:
            st.rerun()
    
    with col2:
        st.header("💡 Exemplos de Perguntas")
        example_queries = [
            "O que fazer para alcançar prosperidade?",
            "Como os Odus orientam sobre dívidas?",
            "Qual Odu fala sobre novos começos?",
            "O que significa fazer ebó?",
            "Quais são os tabus mencionados?"
        ]
        
        for example in example_queries:
            if st.button(f"→ {example}", key=example):
                st.session_state.example_query = example
                st.rerun()
    
    # Processar query de exemplo se existir
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
        search_button = True
    
    # Realizar busca
    if search_button and query:
        with st.spinner("🔍 Buscando nos Odus..."):
            # Busca semântica
            start_time = time.time()
            contexts = st.session_state.indexer.search(query, n_results=n_results)
            search_time = time.time() - start_time
            
        with st.spinner("💭 Interpretando sabedoria ancestral..."):
            # Gerar resposta
            response = st.session_state.llm.generate_response(query, contexts)
            total_time = time.time() - start_time
        
        # Exibir resposta
        st.divider()
        st.header("🎯 Resposta dos Odus")
        
        # Resposta principal
        st.markdown(response['answer'])
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Tempo de resposta", f"{total_time:.2f}s")
        with col2:
            st.metric("📚 Contextos usados", response['contexts_used'])
        with col3:
            st.metric("🔍 Tempo de busca", f"{search_time:.2f}s")
        
        # Contextos encontrados (expansível)
        with st.expander(f"📖 Ver {len(contexts)} trechos encontrados"):
            for i, ctx in enumerate(contexts, 1):
                st.markdown(f"**Trecho {i} - Fonte: {ctx['metadata']['source']}**")
                st.text(ctx['text'][:500] + "..." if len(ctx['text']) > 500 else ctx['text'])
                st.divider()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    🔮 Oluwo IA POC - Preservando a sabedoria Yorubá<br>
    ⚠️ Esta é uma POC. As interpretações devem ser validadas com babalaôs experientes.
    </small>
</div>
""", unsafe_allow_html=True)
3.5 Script de Inicialização (setup.py)
python#!/usr/bin/env python3
"""
Script para configurar o ambiente da POC Oluwo IA
"""

import subprocess
import sys
from pathlib import Path

def setup_environment():
    print("🚀 Configurando Oluwo IA POC...")
    
    # 1. Criar estrutura de pastas
    dirs = [
        "data/odus_pdf",
        "data/processed",
        "data/vector_db",
        "prompts",
        "utils"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Estrutura de pastas criada")
    
    # 2. Instalar dependências
    print("📦 Instalando dependências Python...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # 3. Instalar Ollama
    print("\n⚠️  IMPORTANTE: Instale o Ollama manualmente:")
    print("   Mac/Linux: curl -fsSL https://ollama.com/install.sh | sh")
    print("   Windows: Baixe de https://ollama.com/download")
    
    # 4. Instruções finais
    print("\n" + "="*50)
    print("✅ SETUP CONCLUÍDO!")
    print("="*50)
    print("\nPRÓXIMOS PASSOS:")
    print("1. Coloque os PDFs dos Odus em: data/odus_pdf/")
    print("2. Execute: streamlit run app.py")
    print("3. Use a interface para processar e consultar os Odus")
    print("\nNOTA: Na primeira execução, o Ollama baixará o modelo (~2-4GB)")

if __name__ == "__main__":
    setup_environment()
4. INSTRUÇÕES DE USO DA POC
4.1 Instalação Rápida
bash# 1. Clone ou crie o projeto
mkdir oluwo-ia-poc
cd oluwo-ia-poc

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Instale Ollama (se não tiver)
# Mac/Linux:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: baixe de https://ollama.com/download

# 5. Coloque os PDFs dos Odus em data/odus_pdf/

# 6. Execute a aplicação
streamlit run app.py
4.2 Uso da Interface

Primeira vez:

Clique em "Processar PDFs" na sidebar
Clique em "Criar Índice Vetorial"
Clique em "Inicializar LLM"


Fazer consultas:

Digite sua pergunta
Clique em "Consultar"
Veja a resposta e os contextos usados



5. OTIMIZAÇÕES PARA POC
5.1 Reduzir Uso de Memória
python# Use modelo menor se tiver pouca RAM
model = "phi3:mini"  # Apenas 2GB

# Reduza chunks se necessário
chunk_size = 300  # Menor = mais chunks mas menos memória

# Use FAISS ao invés de ChromaDB se precisar
import faiss
# FAISS é mais leve para POC
5.2 Melhorar Performance
python# Cache simples em memória
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str):
    return indexer.search(query)
6. CUSTOS DA POC
yamlCustos Diretos: R$ 0,00
- LLM: Gratuito (Ollama local)
- Embeddings: Gratuito (sentence-transformers local)
- Vector DB: Gratuito (ChromaDB local)
- Interface: Gratuito (Streamlit local)
- Deploy: Local apenas

Requisitos de Hardware:
- Mínimo: 8GB RAM, 10GB disco
- Recomendado: 16GB RAM, 20GB disco
- GPU: Não necessária (mas acelera)

Tempo de Setup: ~30 minutos
Tempo de Processamento (256 PDFs): ~10-20 minutos
7. PRÓXIMOS PASSOS APÓS POC
Depois de validar o conceito com a POC gratuita:

Versão Beta (Baixo Custo):

Deploy no Hugging Face Spaces (gratuito)
Ou Railway.app ($5/mês)
Use Groq API (gratuita até 30k tokens/dia)


Versão Produção (Quando houver receita):

Migrar para a arquitetura completa original
API robusta com FastAPI
LLM mais poderosa
Deploy em cloud



Esta POC permite validar todo o conceito sem gastar nada! 🎉
