
import pdfplumber
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List

class OduPDFProcessor:
    def __init__(self, input_dir: str = None):
        # Garante caminho absoluto relativo ao arquivo processor.py
        base_dir = Path(__file__).parent
        self.input_dir = base_dir / "data" / "odus_pdf" if input_dir is None else Path(input_dir)
        self.output_dir = base_dir / "data" / "processed"
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
                odu_data["chunks"] = self.create_chunks(full_text, pdf_path.name)
                odu_data["metadata"] = self.extract_metadata(full_text)
        except Exception as e:
            print(f"❌ Erro ao processar {pdf_path.name}: {e}")
        return odu_data

    def create_chunks(self, text: str, filename: str, chunk_size: int = 500) -> List[Dict]:
        """
        Divide o texto em chunks temáticos para indexação, marcando tipo conforme diretrizes:
        - ese_ifa (Yorubá)
        - comentario (tradução/comentário)
        - explicacao (parágrafo objetivo)
        - lista_orisas, lista_tabus, lista_profissoes, etc.
        - tabus (bloco separado)
        """
        chunks = []
        lines = text.split('\n')
        chunk_id = 0
        i = 0
        # Regex para identificar blocos
        ese_ifa_pattern = re.compile(r'^[A-Za-zÀ-ÿ\'\-]+(\s+[A-Za-zÀ-ÿ\'\-]+){1,}$')
        tabu_patterns = [r'Tabus de', r'Nunca deve', r'Não pode', r'Evite']
        explicacao_pattern = re.compile(r'^.{20,}\.$')
        lista_orisas_pattern = re.compile(r'(Orixás|Orisás|Orisas|Òrìṣà)', re.IGNORECASE)
        lista_profissoes_pattern = re.compile(r'(Profissões|Profissoes)', re.IGNORECASE)
        lista_nomes_pattern = re.compile(r'(Nomes)', re.IGNORECASE)
        # Estado
        current_chunk = ""
        current_type = None
        tabus = []
        while i < len(lines):
            line = lines[i].strip()
            # Detectar explicação prática (primeiro parágrafo)
            if i == 0 and explicacao_pattern.match(line):
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id}",
                    "text": line,
                    "type": "explicacao",
                    "metadata": {"source": filename, "chunk_number": chunk_id}
                })
                chunk_id += 1
                i += 1
                continue
            # Detectar esé Ifá em Yorubá (linhas curtas, centralizadas, sem pontuação)
            if ese_ifa_pattern.match(line) and len(line) < 60:
                ese_lines = [line]
                j = i + 1
                while j < len(lines) and ese_ifa_pattern.match(lines[j].strip()) and len(lines[j].strip()) < 60:
                    ese_lines.append(lines[j].strip())
                    j += 1
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id}",
                    "text": '\n'.join(ese_lines),
                    "type": "ese_ifa",
                    "metadata": {"source": filename, "chunk_number": chunk_id}
                })
                chunk_id += 1
                i = j
                # Comentário/tradução logo abaixo
                if i < len(lines) and len(lines[i].strip()) > 20:
                    chunks.append({
                        "chunk_id": f"{filename}_{chunk_id}",
                        "text": lines[i].strip(),
                        "type": "comentario",
                        "metadata": {"source": filename, "chunk_number": chunk_id}
                    })
                    chunk_id += 1
                    i += 1
                continue
            # Detectar listas finais
            if lista_orisas_pattern.search(line):
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id}",
                    "text": line,
                    "type": "lista_orisas",
                    "metadata": {"source": filename, "chunk_number": chunk_id}
                })
                chunk_id += 1
                i += 1
                continue
            if lista_profissoes_pattern.search(line):
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id}",
                    "text": line,
                    "type": "lista_profissoes",
                    "metadata": {"source": filename, "chunk_number": chunk_id}
                })
                chunk_id += 1
                i += 1
                continue
            if lista_nomes_pattern.search(line):
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id}",
                    "text": line,
                    "type": "lista_nomes",
                    "metadata": {"source": filename, "chunk_number": chunk_id}
                })
                chunk_id += 1
                i += 1
                continue
            # Detectar tabus
            if any(re.search(pat, line, re.IGNORECASE) for pat in tabu_patterns):
                # Extrair bloco de tabus
                tabu_block = []
                j = i
                while j < len(lines) and (any(re.search(pat, lines[j], re.IGNORECASE) for pat in tabu_patterns) or len(lines[j].strip()) > 0):
                    if len(lines[j].strip()) > 0:
                        tabu_block.append(lines[j].strip())
                    j += 1
                # Regex para extrair descrição e explicação
                for tabu_line in tabu_block:
                    m = re.match(r'(Nunca deve.*?)(?:,|\.|;|\-|\–)?\s*(para.*)?$', tabu_line)
                    if m:
                        tabus.append({
                            "descricao": m.group(1).strip(),
                            "explicacao": m.group(2).strip() if m.group(2) else ""
                        })
                    else:
                        tabus.append({"descricao": tabu_line, "explicacao": ""})
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id}",
                    "text": '\n'.join(tabu_block),
                    "type": "tabus",
                    "metadata": {"source": filename, "chunk_number": chunk_id}
                })
                chunk_id += 1
                i = j
                continue
            # Chunk genérico se nada bater
            if len(line) > 20:
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id}",
                    "text": line,
                    "type": "generico",
                    "metadata": {"source": filename, "chunk_number": chunk_id}
                })
                chunk_id += 1
            i += 1
        # Adiciona campo tabus no final (sempre presente)
        if not tabus:
            tabus = []
        # Adiciona como chunk separado para garantir presença
        chunks.append({
            "chunk_id": f"{filename}_tabus_final",
            "text": json.dumps(tabus, ensure_ascii=False, indent=2),
            "type": "tabus_final",
            "metadata": {"source": filename, "chunk_number": "tabus_final"}
        })
        return chunks

    def extract_metadata(self, text: str) -> Dict:
        """Extrai metadados básicos do texto"""
        metadata = {
            "orisas": [],
            "ebos": [],
            "keywords": []
        }
        orisa_pattern = r'(Ifá|Orí|Èṣù|Ọbàtálá|Ògún|Ṣàngó|Ọ̀ṣun|Yemọja|Ọya|Òṣọ̀ọ̀sì|Obà|Egúngún|Olókun|Òrìṣà\s?Oko|Ṣọ̀pọ̀na|Osanyìn|Erinlẹ̀|Ajé|Ibeji)'
        ebo_pattern = r'ẹbọ.*?(?:com|para).*?(?:\.|;|\n)'
        metadata["orisas"] = list(set(re.findall(orisa_pattern, text, re.IGNORECASE)))
        metadata["ebos"] = re.findall(ebo_pattern, text, re.IGNORECASE)[:5]
        return metadata

    def generate_id(self, filename: str) -> str:
        """Gera ID único para o Odu"""
        return hashlib.md5(filename.encode()).hexdigest()[:12]

    def process_all_pdfs(self):
        """Processa todos os PDFs na pasta e subpastas, aceitando nomes com caracteres especiais"""
        all_odus = []
        # Busca recursiva por PDFs
        pdf_files = list(self.input_dir.rglob("*.pdf"))
        print(f"🔍 Encontrados {len(pdf_files)} PDFs para processar:")
        for pdf_path in pdf_files:
            print(f" - {pdf_path}")
        for pdf_path in pdf_files:
            odu_data = self.process_single_pdf(pdf_path)
            all_odus.append(odu_data)
            output_file = self.output_dir / f"{odu_data['odu_id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(odu_data, f, ensure_ascii=False, indent=2)
        with open(self.output_dir / "all_odus.json", 'w', encoding='utf-8') as f:
            json.dump(all_odus, f, ensure_ascii=False, indent=2)
        print(f"✅ Processamento concluído! {len(all_odus)} Odus processados")
        return all_odus

if __name__ == "__main__":
    OduPDFProcessor().process_all_pdfs()