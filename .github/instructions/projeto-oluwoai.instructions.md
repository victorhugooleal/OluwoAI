
# Instruções do Projeto Oluwo AI

## 1. Visão Geral do Projeto
**Objetivo:**
Desenvolver um agente de IA especializado chamado "Oluwo IA" capaz de interpretar e fornecer orientações baseadas nos 256 Odus de Ifá, preservando a sabedoria ancestral Yorubá e tornando-a acessível através de tecnologia moderna.

**Funcionalidades Principais:**
- Busca semântica em todos os 256 Odus
- Respostas contextualizadas com referências específicas
- Interpretação multilíngue (Yorubá/Português)
- Rastreabilidade completa do raciocínio
- Preservação da autenticidade cultural

---

## 2. Arquitetura Técnica
**Stack Tecnológico Recomendado:**
- Python 3.11+
- FastAPI
- LLM primária: GPT-4 ou Claude 3 Opus
- LLM secundária: Llama 3.1 70B (backup local)
- Banco vetorial: Qdrant ou Weaviate
- Processamento PDF: Docling + PyMuPDF
- Embedding: text-embedding-3-large (OpenAI) ou multilingual-e5-large
- Cache: Redis
- Monitoramento: Langfuse
- Deploy: Docker + Kubernetes

**Arquitetura RAG (Retrieval-Augmented Generation):**
```python
class OluwoRAGPipeline:
    """
    Pipeline completo para processamento de consultas sobre Odus
    """
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.retriever = SemanticRetriever()
        self.reranker = CrossEncoderReranker()
        self.llm = LLMOrchestrator()
        self.response_formatter = ResponseFormatter()
```

---

## 3. Pipeline de Processamento de Documentos
**Conversão e Estruturação dos PDFs:**
Estrutura JSON para cada Odu:
```json
{
  "odu_id": "oyeku_ika",
  "odu_number": 63,
  "nome_principal": "Òyẹ̀kú Ìká",
  "nomes_alternativos": ["Oyeku-Ika", "Oyeku Ika"],
  "capitulo": "11",
  "versos": [
    {
      "verso_id": "v1",
      "tipo": "ese_ifa",
      "texto_yoruba": "Ifá diz que esta pessoa está em dívida...",
      "texto_portugues": "...",
      "interpretacao": "...",
      "ebo_recomendado": {
        "materiais": ["dois galos", "duas galinhas", "dinheiro"],
        "proposito": "limpar dívidas"
      },
      "palavras_chave": ["dívida", "ẹbọ", "limpeza espiritual"]
    }
  ],
  "orisas_relacionados": ["Ifá", "Orí", "Èṣù Òdàrà", "Ọbàtálá"],
  "tabus": [],
  "profissoes_indicadas": [],
  "nomes_recomendados": {
    "masculinos": [],
    "femininos": []
  },
  "metadata": {
    "data_processamento": "2024-01-01",
    "fonte": "arquivo.pdf",
    "paginas": [80, 85]
  }
}
```

Exemplo de script de processamento com Docling:
```python
import docling
from pathlib import Path
import json
import re
from typing import Dict, List

class OduDocumentProcessor:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.docling_converter = docling.DocumentConverter()
    
    def process_all_odus(self):
        """Processa todos os 256 PDFs de Odus"""
        odus_data = []
        for pdf_file in self.input_dir.glob("*.pdf"):
            print(f"Processando: {pdf_file.name}")
            markdown_content = self.docling_converter.convert(
                pdf_file,
                output_format="markdown"
            )
            odu_structured = self.parse_odu_content(markdown_content)
            odu_structured['metadata'] = {
                'source_file': pdf_file.name,
                'processing_date': datetime.now().isoformat()
            }
            odus_data.append(odu_structured)
        with open(self.output_dir / "odus_complete.json", "w") as f:
            json.dump(odus_data, f, ensure_ascii=False, indent=2)
    
    def parse_odu_content(self, content: str) -> Dict:
        """Parser específico para estrutura dos Odus"""
        # Implementar regex patterns para extrair:
        # - Versos de Ifá
        # - Interpretações
        # - Ebós recomendados
        # - Orixás mencionados
        # - Tabus e recomendações
        pass
```

---

## 4. Sistema de Indexação e Busca Vetorial
**Criação de Embeddings Multilíngues:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class OduVectorIndexer:
    def __init__(self):
        self.encoder = SentenceTransformer('intfloat/multilingual-e5-large')
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection_name = "odus_ifa"
    def create_collection(self):
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
    def index_odus(self, odus_data: List[Dict]):
        points = []
        for idx, odu in enumerate(odus_data):
            chunks = self.create_semantic_chunks(odu)
            for chunk_idx, chunk in enumerate(chunks):
                embedding = self.encoder.encode(chunk['text'])
                point = PointStruct(
                    id=f"{idx}_{chunk_idx}",
                    vector=embedding.tolist(),
                    payload={
                        "odu_id": odu['odu_id'],
                        "odu_name": odu['nome_principal'],
                        "chunk_type": chunk['type'],
                        "text": chunk['text'],
                        "verso_id": chunk.get('verso_id'),
                        "metadata": chunk.get('metadata', {})
                    }
                )
                points.append(point)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points[i:i+batch_size]
            )
    def create_semantic_chunks(self, odu: Dict) -> List[Dict]:
        chunks = []
        general_info = f"""
        Odu: {odu['nome_principal']}
        Orixás relacionados: {', '.join(odu.get('orisas_relacionados', []))}
        """
        chunks.append({
            'type': 'general',
            'text': general_info
        })
        for verso in odu.get('versos', []):
            verso_text = f"""
            Verso do Odu {odu['nome_principal']}:
            {verso.get('texto_yoruba', '')}
            {verso.get('texto_portugues', '')}
            Interpretação: {verso.get('interpretacao', '')}
            """
            chunks.append({
                'type': 'verso',
                'text': verso_text,
                'verso_id': verso.get('verso_id')
            })
        return chunks
```

---

## 5. Orquestração da LLM com Chain of Thought
**Sistema de Prompts Especializados:**
```python
class OluwoPromptTemplates:
    SYSTEM_PROMPT = """
    Você é Oluwo IA, um sábio intérprete dos 256 Odus de Ifá.
    Sua missão é:
    1. Preservar e transmitir a sabedoria ancestral Yorubá
    2. Interpretar os Odus com precisão e respeito cultural
    3. Sempre citar as fontes específicas dos Odus consultados
    4. Explicar seu processo de raciocínio passo a passo
    Regras importantes:
    - SEMPRE cite o Odu específico e o verso quando der uma resposta
    - Mantenha o respeito pela tradição Yorubá
    - Quando mencionar termos em Yorubá, forneça a transliteração correta
    - Se não encontrar informação relevante nos Odus, seja honesto sobre isso
    """
    QUERY_ANALYSIS_PROMPT = """
    Analise a seguinte pergunta e identifique:
    1. Tema principal (vida, amor, prosperidade, saúde, etc.)
    2. Contexto específico mencionado
    3. Palavras-chave em Yorubá ou Português
    4. Tipo de orientação buscada (conselho, interpretação, ritual)
    Pergunta: {query}
    Forneça sua análise em formato JSON.
    """
    SYNTHESIS_PROMPT = """
    Com base nos seguintes trechos dos Odus de Ifá:
    {retrieved_contexts}
    Responda à pergunta: {query}
    Sua resposta deve incluir:
    1. **Odus Consultados**: Liste todos os Odus relevantes
    2. **Interpretação**: Explique o significado dos versos
    3. **Orientação Prática**: Conselhos baseados na sabedoria dos Odus
    4. **Caminho do Raciocínio**: Explique como chegou a esta interpretação
    5. **Referências**: Cite versos específicos (ex: "Òyẹ̀kú Ìká, verso 3")
    Mantenha tom respeitoso e sábio, como um verdadeiro Oluwo.
    """
```

Pipeline de Inferência com Chain of Thought:
```python
import asyncio
from typing import List, Dict
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import BaseMessage

class OluwoInferencePipeline:
    def __init__(self, llm_client, vector_store, reranker):
        self.llm = llm_client
        self.vector_store = vector_store
        self.reranker = reranker
        self.callback_handler = AsyncIteratorCallbackHandler()
    async def process_query(self, user_query: str) -> Dict:
        query_analysis = await self.analyze_query(user_query)
        retrieved_docs = await self.semantic_search(
            query=user_query,
            filters=query_analysis.get('filters', {}),
            top_k=20
        )
        reranked_docs = self.reranker.rerank(
            query=user_query,
            documents=retrieved_docs,
            top_k=5
        )
        response = await self.generate_response(
            query=user_query,
            contexts=reranked_docs,
            analysis=query_analysis
        )
        final_response = self.format_response_with_citations(
            response=response,
            sources=reranked_docs,
            reasoning_path=self.build_reasoning_path(query_analysis, reranked_docs)
        )
        return final_response
    def build_reasoning_path(self, analysis: Dict, docs: List) -> str:
        path = f"""
        📍 **Caminho do Raciocínio**:
        1. **Análise da Pergunta**: 
           - Tema identificado: {analysis.get('theme')}
           - Contexto: {analysis.get('context')}
        2. **Odus Consultados**:
           {self.format_consulted_odus(docs)}
        3. **Versos Relevantes Identificados**:
           {self.format_relevant_verses(docs)}
        4. **Síntese da Sabedoria**:
           Combinei os ensinamentos destes Odus para formar a orientação
        """
        return path
```

---

## 6. Sistema de Cache e Otimização
**Cache Inteligente com Redis:**
```python
import redis
import hashlib
import json
from datetime import timedelta

class OluwoCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.default_ttl = timedelta(hours=24)
    def get_or_compute(self, key: str, compute_func, ttl=None):
        cache_key = self.generate_cache_key(key)
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        result = compute_func()
        self.redis.setex(
            cache_key,
            ttl or self.default_ttl,
            json.dumps(result, ensure_ascii=False)
        )
        return result
    def generate_cache_key(self, query: str) -> str:
        return f"oluwo:query:{hashlib.md5(query.encode()).hexdigest()}"
```

---

## 7. API REST com FastAPI
**Endpoints Principais:**
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Oluwo IA API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None
    include_reasoning: bool = True
    max_odus: int = 3

class OduReference(BaseModel):
    odu_name: str
    verso_id: str
    relevance_score: float
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    odus_consulted: List[OduReference]
    reasoning_path: Optional[str]
    confidence_score: float
    processing_time_ms: int

@app.post("/api/v1/consult", response_model=QueryResponse)
async def consult_odus(request: QueryRequest):
    try:
        start_time = time.time()
        result = await oluwo_pipeline.process_query(
            user_query=request.question,
            context=request.context
        )
        response = QueryResponse(
            answer=result['answer'],
            odus_consulted=result['sources'],
            reasoning_path=result['reasoning'] if request.include_reasoning else None,
            confidence_score=result['confidence'],
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/odu/{odu_id}")
async def get_odu_details(odu_id: str):
    # Implementar busca do Odu
    pass

@app.post("/api/v1/index/refresh")
async def refresh_index(background_tasks: BackgroundTasks):
    background_tasks.add_task(reindex_all_odus)
    return {"message": "Reindexação iniciada em background"}
```

---

## 8. Monitoramento e Observabilidade
**Sistema de Logs e Métricas:**
```python
from langfuse import Langfuse
from prometheus_client import Counter, Histogram, Gauge
import structlog

langfuse = Langfuse(
    public_key="your-public-key",
    secret_key="your-secret-key"
)
query_counter = Counter('oluwo_queries_total', 'Total de consultas')
query_duration = Histogram('oluwo_query_duration_seconds', 'Duração das consultas')
active_users = Gauge('oluwo_active_users', 'Usuários ativos')
logger = structlog.get_logger()

class OluwoMonitoring:
    @staticmethod
    def trace_llm_call(func):
        def wrapper(*args, **kwargs):
            with langfuse.trace(name=func.__name__) as trace:
                result = func(*args, **kwargs)
                trace.log(
                    input=kwargs.get('prompt'),
                    output=result,
                    metadata={
                        'model': kwargs.get('model', 'gpt-4'),
                        'temperature': kwargs.get('temperature', 0.7)
                    }
                )
            return result
        return wrapper
```

---

## 9. Testes e Validação
**Suite de Testes:**
```python
import pytest
from unittest.mock import Mock, patch

class TestOluwoSystem:
    @pytest.fixture
    def sample_odu(self):
        return {
            "odu_id": "oyeku_ika",
            "nome_principal": "Òyẹ̀kú Ìká",
            "versos": []
        }
    def test_document_processing(self, sample_odu):
        processor = OduDocumentProcessor()
        result = processor.parse_odu_content(sample_odu)
        assert result['odu_id'] == "oyeku_ika"
        assert len(result['versos']) > 0
    def test_semantic_search(self):
        query = "Como alcançar prosperidade?"
        results = vector_store.search(query, top_k=5)
        assert len(results) <= 5
        assert all(r.score >= 0.7 for r in results)
    def test_response_generation(self):
        query = "Qual Odu fala sobre dívidas?"
        response = oluwo_pipeline.process_query(query)
        assert "Òyẹ̀kú Ìká" in response['answer']
        assert len(response['odus_consulted']) > 0
        assert response['confidence_score'] > 0.8
```

---

## 10. Deployment e Infraestrutura
**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/models
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  oluwo-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - redis
      - qdrant
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - oluwo-api
volumes:
  redis-data:
  qdrant-data:
```

---

## 11. Instruções de Implementação Passo a Passo
**Fase 1: Preparação de Dados (Semana 1)**
- Instalar Docling e dependências
- Processar todos os 256 PDFs para JSON estruturado
- Validar qualidade da extração
- Criar backup dos dados processados

**Fase 2: Sistema de Indexação (Semana 2)**
- Configurar Qdrant localmente
- Implementar chunking strategy
- Gerar embeddings multilíngues
- Indexar todos os Odus
- Testar qualidade da busca

**Fase 3: Pipeline RAG (Semana 3)**
- Implementar retriever semântico
- Configurar reranking
- Integrar LLM (GPT-4 ou Claude)
- Implementar chain of thought
- Adicionar sistema de citações

**Fase 4: API e Interface (Semana 4)**
- Desenvolver API REST com FastAPI
- Implementar cache com Redis
- Adicionar autenticação JWT
- Criar documentação Swagger
- Implementar rate limiting

**Fase 5: Testes e Otimização (Semana 5)**
- Criar suite de testes automatizados
- Realizar testes de carga
- Otimizar performance
- Implementar monitoring
- Documentar edge cases

**Fase 6: Deploy e Manutenção (Semana 6)**
- Containerizar aplicação
- Configurar CI/CD
- Deploy em cloud (AWS/GCP/Azure)
- Configurar backups automáticos
- Estabelecer SLAs

---

## 12. Considerações Finais
**Requisitos de Hardware Mínimos:**
- CPU: 8 cores
- RAM: 32GB
- Storage: 100GB SSD
- GPU: Opcional (para embeddings locais)

**Custos Estimados (Mensal):**
- LLM API: $500-2000 (baseado em volume)
- Infraestrutura Cloud: $200-500
- Vector Database: $100-300
- Total: ~$800-2800/mês

**Métricas de Sucesso:**
- Precisão de busca: >90%
- Tempo de resposta: <3 segundos
- Disponibilidade: 99.9%
- Satisfação do usuário: >4.5/5