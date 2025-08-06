# # vectorstore.py
# import os
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
# from config import UPLOAD_DIR, PERSIST_DIR, EMBED_MODEL
# from io_utils import file_text, split_to_documents, content_sha256

# def get_vectordb() -> Chroma:
#     embeddings = OllamaEmbeddings(model=EMBED_MODEL)
#     return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# def current_upload_files() -> dict[str, str]:
#     os.makedirs(UPLOAD_DIR, exist_ok=True)
#     files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR)
#              if f.lower().endswith((".pdf", ".docx"))]
#     mapping = {}
#     for p in files:
#         try:
#             mapping[content_sha256(p)] = p  # content-based id
#         except Exception:
#             continue
#     return mapping

# def vectordb_file_ids(vdb: Chroma) -> set[str]:
#     fids, offset, page = set(), 0, 5000
#     while True:
#         results = vdb.get(include=["metadatas"], limit=page, offset=offset)
#         metas = results.get("metadatas", []) or []
#         if not metas: break
#         for m in metas:
#             fid = m.get("file_id")
#             if isinstance(fid, str) and fid: fids.add(fid)
#         if len(metas) < page: break
#         offset += page
#     return fids

# def delete_by_file_id(vdb: Chroma, file_id: str):
#     vdb.delete(where={"file_id": {"$eq": file_id}})

# def add_file_to_vectordb(vdb: Chroma, file_path: str, file_id: str):
#     text = file_text(file_path)
#     if not text.strip(): return
#     meta = {"source": file_path, "file_name": os.path.basename(file_path), "file_id": file_id}
#     docs = split_to_documents(text, meta)
#     vdb.add_documents(docs)

# def sync_vectordb() -> Chroma:
#     vdb = get_vectordb()
#     current_map = current_upload_files()         # {file_id: path}
#     current_ids = set(current_map.keys())
#     existing_ids = vectordb_file_ids(vdb)

#     # delete removed
#     for orphan in (existing_ids - current_ids):
#         delete_by_file_id(vdb, orphan)
#     # add new
#     for new_id in (current_ids - existing_ids):
#         add_file_to_vectordb(vdb, current_map[new_id], new_id)

#     try: vdb.persist()
#     except Exception: pass
#     return vdb
