from abc import ABC, abstractmethod
from typing import Optional, List
import chess
import torch
import re
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from typing import Optional, List, Dict
import json

class Player(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_move(self, fen: str) -> Optional[str]:
        pass


class ChessTransformer(nn.Module):
    def __init__(
        self,
        num_moves,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        self.square_embed = nn.Embedding(13, d_model)    
        self.pos_embed = nn.Embedding(64, d_model)
        self.side_embed = nn.Embedding(2, d_model)
        self.castling_embed = nn.Embedding(2, d_model)
        self.ep_embed = nn.Embedding(65, d_model)         

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # extra state tokens: side, 4 castling, ep
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.norm = nn.LayerNorm(d_model)

        self.policy_head = nn.Linear(d_model, num_moves)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),   # output in [-1,1]
        )

    def forward(self, squares, side_to_move, castling, ep_square):
        B = squares.size(0)
        device = squares.device

        pos_ids = torch.arange(64, device=device).unsqueeze(0).expand(B, 64)

        x = self.square_embed(squares) + self.pos_embed(pos_ids)

        side_tok = self.side_embed(side_to_move).unsqueeze(1)              # [B,1,D]
        castling_tok = self.castling_embed(castling)                       # [B,4,D]
        ep_tok = self.ep_embed(ep_square).unsqueeze(1)                     # [B,1,D]
        cls_tok = self.cls_token.expand(B, 1, -1)                          # [B,1,D]

        x = torch.cat([cls_tok, side_tok, castling_tok, ep_tok, x], dim=1) # [B,71,D]
        x = self.encoder(x)
        x = self.norm(x)

        pooled = x[:, 0]   # CLS token

        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)

        return policy_logits, value


class TransformerPlayer(Player):

    UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)

    PIECE_VALUE = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    PIECE_TO_ID = {
        None: 0,
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
        "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
    }

    def __init__(
        self,
        name: str,
        repo_id: str = "Rients-uu/chess-transformer",
        model_filename: str = "chess_transformer.pt",
        move_vocab_filename: str = "move_to_id.json",
        alpha: float = 5.0,
        beta: float = 2.0,
        debug: bool = False,
    ):
        super().__init__(name)
        self.repo_id = repo_id
        self.model_filename = model_filename
        self.move_vocab_filename = move_vocab_filename
        self.model_path = None
        self.move_vocab_path = None
        self.alpha = alpha
        self.beta = beta
        self.debug = debug

        self._model = None
        self._model_unavailable = False
        self.move_to_id: Dict[str, int] = {}
        self.id_to_move: Dict[int, str] = {}

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    # ---------------------------
    # Model loading
    # ---------------------------
    def _ensure_model(self) -> bool:
        if self._model_unavailable:
            return False
        if self._model is not None and self.move_to_id:
            return True

        try:
            # Download files from Hugging Face Hub
            self.model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.model_filename,
                repo_type="model",
            )
            self.move_vocab_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.move_vocab_filename,
                repo_type="model",
            )

            with open(self.move_vocab_path, "r") as f:
                self.move_to_id = json.load(f)

            self.id_to_move = {idx: move for move, idx in self.move_to_id.items()}

            self._model = ChessTransformer(num_moves=len(self.move_to_id))
            self._model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self._model.to(self.device)
            self._model.eval()
            return True

        except Exception as e:
            print("MODEL LOAD FAILED:", repr(e))
            self._model_unavailable = True
            self._model = None
            self.move_to_id = {}
            self.id_to_move = {}
            return False

    # ---------------------------
    # FEN -> tensor features
    # ---------------------------
    def _fen_to_features(self, fen: str):
        board = chess.Board(fen)

        squares = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            squares.append(self.PIECE_TO_ID[piece.symbol()] if piece else 0)

        side_to_move = 1 if board.turn == chess.WHITE else 0

        castling = [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        ]

        ep_square = board.ep_square if board.ep_square is not None else 64

        return {
            "squares": torch.tensor(squares, dtype=torch.long),
            "side_to_move": torch.tensor(side_to_move, dtype=torch.long),
            "castling": torch.tensor(castling, dtype=torch.long),
            "ep_square": torch.tensor(ep_square, dtype=torch.long),
        }

    # ---------------------------
    # Model scoring
    # ---------------------------
    def _score_legal_moves(self, fen: str, legal_uci: List[str]) -> Dict[str, float]:
        feats = self._fen_to_features(fen)

        batch = {
            "squares": feats["squares"].unsqueeze(0).to(self.device),
            "side_to_move": feats["side_to_move"].unsqueeze(0).to(self.device),
            "castling": feats["castling"].unsqueeze(0).to(self.device),
            "ep_square": feats["ep_square"].unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            policy_logits, value_pred = self._model(
                batch["squares"],
                batch["side_to_move"],
                batch["castling"],
                batch["ep_square"],
            )

        logits = policy_logits[0]  # shape: [num_moves]

        legal_scores = {}
        for mv in legal_uci:
            idx = self.move_to_id.get(mv)
            if idx is not None and idx < logits.shape[0]:
                legal_scores[mv] = logits[idx].item()

        return legal_scores

    # ---------------------------
    # Heuristic scoring
    # ---------------------------
    def _captured_piece_value(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_en_passant(move):
            return self.PIECE_VALUE[chess.PAWN]

        captured = board.piece_at(move.to_square)
        if not captured:
            return 0
        return self.PIECE_VALUE[captured.piece_type]

    def heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        mover = board.piece_at(move.from_square)
        mover_value = self.PIECE_VALUE[mover.piece_type] if mover else 0

        if board.is_capture(move):
            captured_value = self._captured_piece_value(board, move)
            score += 10 * captured_value - mover_value

        if move.promotion:
            score += self.PIECE_VALUE.get(move.promotion, 0) * 10

        board.push(move)
        if board.is_checkmate():
            score += 100000
        elif board.is_check():
            score += 1
        board.pop()

        file_idx = chess.square_file(move.to_square)
        rank_idx = chess.square_rank(move.to_square)
        score += 3 - abs(3.5 - file_idx)
        score += 3 - abs(3.5 - rank_idx)

        return score

    def _fallback_best_legal(self, board: chess.Board, legal_moves: List[chess.Move]) -> str:
        best_score = -10**9
        best_move = legal_moves[0]

        for move in legal_moves:
            score = self.heuristic_score(board, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci()

    # ---------------------------
    # Hybrid selection
    # ---------------------------
    def best_legal_move(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        legal_uci = [mv.uci() for mv in legal_moves]

        legal_scores = self._score_legal_moves(fen, legal_uci)

        if not legal_scores:
            return self._fallback_best_legal(board, legal_moves)

        filtered_moves = []
        model_scores = []
        heuristic_scores = []

        for mv in legal_moves:
            uci = mv.uci()
            if uci not in legal_scores:
                continue

            filtered_moves.append(mv)
            model_scores.append(legal_scores[uci])
            heuristic_scores.append(self.heuristic_score(board, mv))

        if not filtered_moves:
            return self._fallback_best_legal(board, legal_moves)

        # Normalize model scores
        m_min = min(model_scores)
        m_max = max(model_scores)
        if m_max > m_min:
            norm_model_scores = [(s - m_min) / (m_max - m_min) for s in model_scores]
        else:
            norm_model_scores = [0.0 for _ in model_scores]

        # Normalize heuristic scores
        h_min = min(heuristic_scores)
        h_max = max(heuristic_scores)
        if h_max > h_min:
            norm_heuristic_scores = [(s - h_min) / (h_max - h_min) for s in heuristic_scores]
        else:
            norm_heuristic_scores = [0.0 for _ in heuristic_scores]

        best_idx = 0
        best_score = float("-inf")

        for i, mv in enumerate(filtered_moves):
            final_score = (
                self.alpha * norm_model_scores[i]
                + self.beta * norm_heuristic_scores[i]
            )

            if self.debug:
                print(
                    f"{mv.uci():6s} | "
                    f"model={model_scores[i]:9.4f} | "
                    f"norm_model={norm_model_scores[i]:6.3f} | "
                    f"heuristic={heuristic_scores[i]:9.3f} | "
                    f"norm_heur={norm_heuristic_scores[i]:6.3f} | "
                    f"final={final_score:8.4f}"
                )

            if final_score > best_score:
                best_score = final_score
                best_idx = i

        return filtered_moves[best_idx].uci()

    # ---------------------------
    # Get move function
    # ---------------------------
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        if self._ensure_model():
            try:
                return self.best_legal_move(fen)
            except Exception as e:
                print("SCORING FAILED:", repr(e))

        return self._fallback_best_legal(board, legal_moves)
    

