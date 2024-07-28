from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Collection
from PIL.Image import Image
import chess
from chess import Board, PieceType
from chess.pgn import Game
import re
from collections import defaultdict
from itertools import chain


"""Classes"""


@dataclass
class YoloBbox:
    x: float
    """The x-coordinate of the center of the bounding box relative to the image width."""
    y: float
    """The y-coordinate of the center of the bounding box relative to the image height."""
    width: float
    """Width of the bounding box relative to the image width."""
    height: float
    """Height of the bounding box relative to the image height."""

    image_height: int
    """Height of the image in pixels."""
    image_width: int
    """Width of the image in pixels."""

    @cached_property
    def lurl(self) -> tuple[float, float, float, float]:
        """
        Get a bounding box in the format `(left, upper, right, lower)` in pixel coordinates.
        """
        return (
            (self.x - self.width / 2) * self.image_width,
            (self.y - self.height / 2) * self.image_height,
            (self.x + self.width / 2) * self.image_width,
            (self.y + self.height / 2) * self.image_height,
        )

    @staticmethod
    def has_overlap(bbox1: "YoloBbox", bbox2: "YoloBbox") -> bool:
        """Check if two bounding boxes overlap."""
        left_1, upper_1, right_1, lower_1 = bbox1.lurl
        left_2, upper_2, right_2, lower_2 = bbox2.lurl

        return not (
            right_1 < left_2
            or right_2 < left_1
            or lower_1 < upper_2
            or lower_2 < upper_1
        )

    def has_overlap(self, other: "YoloBbox") -> bool:
        """Check if this bounding box overlaps with another bounding box."""
        return YoloBbox.has_overlap(self, other)

    @staticmethod
    def overlap_area(bbox1: "YoloBbox", bbox2: "YoloBbox") -> float:
        """Calculate the area of overlap between two bounding boxes. Returns 0 if there is no overlap."""
        if not YoloBbox.has_overlap(bbox1, bbox2):
            return 0

        left_1, upper_1, right_1, lower_1 = bbox1.lurl
        left_2, upper_2, right_2, lower_2 = bbox2.lurl

        left = max(left_1, left_2)
        upper = max(upper_1, upper_2)
        right = min(right_1, right_2)
        lower = min(lower_1, lower_2)

        return (right - left) * (lower - upper)

    def overlap_area(self, other: "YoloBbox") -> float:
        """Calculate the area of overlap between this bounding box and another bounding box. Returns 0 if there is no overlap."""
        return YoloBbox.overlap_area(self, other)

    @staticmethod
    def idx_of_bbox_with_max_overlap(
        target_bbox: "YoloBbox",
        bboxes: list["YoloBbox"],
    ) -> tuple[int | None, float]:
        """
        Get the index of the bounding box in `bboxes` that has the maximum overlap with `target_bbox`,
        along with the area of the overlap. If no bounding box has any overlap, returns (`None`, 0).
        """
        max_overlap_area = 0
        max_overlap_idx = None

        for i, bbox in enumerate(bboxes):
            if not YoloBbox.has_overlap(bbox, target_bbox):
                continue

            overlap_area = YoloBbox.overlap_area(bbox, target_bbox)

            if max_overlap_area is None or overlap_area > max_overlap_area:
                max_overlap_area = overlap_area
                max_overlap_idx = i

        return max_overlap_idx, max_overlap_area

    def idx_of_bbox_with_max_overlap(
        self, bboxes: list["YoloBbox"]
    ) -> tuple[int | None, float]:
        """
        Get the index of the bounding box in `bboxes` that has the maximum overlap with this bounding box,
        along with the area of the overlap. If no bounding box has any overlap, returns (`None`, 0).
        """
        return YoloBbox.idx_of_bbox_with_max_overlap(bboxes, self)


class SanParts:
    """Store the parts of a SAN move from a REGEX match. This does not handle castling moves."""

    SAN_REGEX = re.compile(
        r"^([NBKRQ])?([a-h])?([1-8])?([\-x])?([a-h][1-8])(=?[nbrqkNBRQK])?([\+#])?\Z"
    )  # Note: Added more capture groups here than in `chess.SAN_REGEX`

    def __init__(
        self,
        piece: str | None,
        file: str | None,
        rank: str | None,
        capture: str | None,
        to_square: str,
        promotion: str | None,
        check: str | None,
    ):
        self.piece = piece
        self.file = file
        self.rank = rank
        self.capture = capture
        self.to_square = to_square
        self.promotion = promotion
        self.check = check

    @classmethod
    def from_str(cls, san: str) -> "SanParts":
        """
        Parse a SAN string into its individual components using the `SAN_REGEX` pattern.
        """
        match = cls.SAN_REGEX.match(san)
        assert match is not None, f"Failed to match SAN: {san}"
        return cls(*match.groups())

    def __iter__(self):
        return iter(
            [
                self.piece,
                self.file,
                self.rank,
                self.capture,
                self.to_square,
                self.promotion,
                self.check,
            ]
        )

    def __str__(self) -> str:
        """
        Reconstruct the SAN from the individual components of its REGEX match.
        """

        piece = "" if self.piece is None else self.piece
        file = "" if self.file is None else self.file
        rank = "" if self.rank is None else self.rank
        to_square = self.to_square
        capture = "" if self.capture is None else self.capture
        promotion = "" if self.promotion is None else self.promotion
        check = "" if self.check is None else self.check
        return f"{piece}{file}{rank}{capture}{to_square}{promotion}{check}"

    def copy(self) -> "SanParts":
        return SanParts(*self)


"""Utility functions"""


def crop_to_bbox(image: Image, bbox: YoloBbox) -> Image:
    """Crop an image to a specified bounding box."""
    left, upper, right, lower = bbox.lurl
    return image.crop((left, upper, right, lower))


def moves_where_all_conditions_hold(
    moves: Collection[chess.Move],
    conditions: Collection[Callable[[chess.Move], bool]],
) -> set[chess.Move]:
    """
    Return the moves in `moves` that satisfy all the conditions in `conditions`.
    """
    output = set()
    for move in moves:
        if all(cond(move) for cond in conditions):
            output.add(move)
    return output


def get_sans_to_pseudo_sans_map(board: Board) -> dict[str, list[str]]:
    """
    Given a board, generate the legal move list as SAN strings, and for each string,
    generate different pseudo-SAN strings that a human might write on a notation sheet
    instead of the "correct" SAN.

    For example, if "Bxc6" taking a knight is a legal move, also add:
      - "BxN"
      - "b5c6"
      - "Bx" (if it's the only legal bishop capture)
      - "xc6" (if no other piece can take on c6)
      - etc.

    The returned dict maps each valid SAN to a `list` of **globally unique**
    SANs/pseudo-SANs that could represent it (including the valid SAN itself).
    This means all the elements in the values (lists) of the returned dict are
    guaranteed to be unique so there won't be any ambiguity when mapping back
    from a pseudo-SAN to a SAN.
    """

    legal_moves = list(board.legal_moves)
    sans: defaultdict[str, list] = defaultdict(list)

    def _append_if_unique(san: str, new_san: str) -> bool:
        if new_san not in sans[san]:
            sans[san].append(new_san)
            return True
        return False

    def _append_pseudo_san(
        san: str,
        pseudo_san_parts: SanParts,
        *,
        and_without_check: bool = True,
        and_without_capture: bool = True,
        and_with_inaccuracy: bool = True,
        and_with_interesting: bool = True,
        and_with_mistake: bool = True,
        and_with_blunder: bool = True,
    ):
        _append_if_unique(san, str(pseudo_san_parts))

        # Without check/mate
        if and_without_check:
            new_pseudo_san_parts = pseudo_san_parts.copy()
            new_pseudo_san_parts.check = ""
            _append_if_unique(san, str(new_pseudo_san_parts))

        # Without capture
        if and_without_capture:
            new_pseudo_san_parts = pseudo_san_parts.copy()
            new_pseudo_san_parts.capture = ""
            _append_if_unique(san, str(new_pseudo_san_parts))

        # Without check/mate or capture
        if and_without_check and and_without_capture:
            new_pseudo_san_parts = pseudo_san_parts.copy()
            new_pseudo_san_parts.check = ""
            new_pseudo_san_parts.capture = ""
            _append_if_unique(san, str(new_pseudo_san_parts))

        # With inaccuracy
        if and_with_inaccuracy:
            _append_if_unique(san, str(pseudo_san_parts) + "!?")

        # With interesting
        if and_with_interesting:
            _append_if_unique(san, str(pseudo_san_parts) + "!")

        # With mistake
        if and_with_mistake:
            _append_if_unique(san, str(pseudo_san_parts) + "?")

        # With blunder
        if and_with_blunder:
            _append_if_unique(san, str(pseudo_san_parts) + "??")

    # Add the regular SANs first
    for move in legal_moves:
        san = board.san(move)

        # This should always be unique here
        assert _append_if_unique(san, san)

        # It's a bit harder to apply the logic below to
        # castling moves, let's just handle them manually
        if board.is_castling(move):
            # 'O-O+' -> 'O-O'
            _append_if_unique(san, san.replace("+", ""))
            # 'O-O#' -> 'O-O'
            _append_if_unique(san, san.replace("#", ""))
            continue

        # UCI move
        uci = move.uci()
        if move.promotion is not None:
            assert len(uci) == 5
            from_file, from_rank = uci[:2]
            to_square = uci[2:4]
            promotion = uci[4]

            check = san[-1] if board.gives_check(move) else None
            assert check in (None, "+", "#")

            uci_parts = SanParts(
                None,
                from_file,
                from_rank,
                "x" if board.is_capture(move) else None,
                to_square,
                promotion,
                check,
            )
            _append_pseudo_san(san, uci_parts)
        else:
            assert len(uci) == 4
            from_file, from_rank = uci[:2]
            to_square = uci[2:4]

            check = san[-1] if board.gives_check(move) else None
            assert check in (None, "+", "#")

            uci_parts = SanParts(
                None,
                from_file,
                from_rank,
                "x" if board.is_capture(move) else None,
                to_square,
                None,
                check,
            )
            _append_pseudo_san(san, uci_parts)

    # Add the pseudo-SANs
    for move in legal_moves:
        san = board.san(move)

        if board.is_castling(move):
            # I don't think there could be any pseudo-SANs for castling. If there are,
            # `SAN_REGEX` will need to be updated (doesn't support castling moves yet).
            continue

        # Get the parts of the SAN so we can replace different parts as needed for pseudo-SANs
        san_parts = SanParts.from_str(san)

        if board.is_capture(move):
            capturing_piece_type: PieceType = board.piece_type_at(move.from_square)
            captured_piece_type: PieceType = board.piece_type_at(move.to_square)
            captured_piece_symbol = chess.piece_symbol(captured_piece_type).upper()

            # Test if there is only one way for this piece type to capture the other piece type
            conditions = [
                lambda m: board.piece_type_at(m.from_square) == capturing_piece_type,
                lambda m: board.piece_type_at(m.to_square) == captured_piece_type,
            ]
            if len(moves_where_all_conditions_hold(legal_moves, conditions)) == 1:
                # This is the only way this piece type can capture the other piece type
                # Ex. "BxP", "BxR"
                pseudo_san_parts = san_parts.copy()
                pseudo_san_parts.to_square = captured_piece_symbol
                _append_pseudo_san(san, pseudo_san_parts, and_without_capture=False)

            # Test if this piece type can only make one capture
            conditions = [
                lambda m: board.is_capture(m),
                lambda m: board.piece_type_at(m.from_square) == capturing_piece_type,
            ]
            if len(moves_where_all_conditions_hold(legal_moves, conditions)) == 1:
                # This is the only capture that can be made by this piece type
                # Ex. "Bx", "Rx"
                pseudo_san_parts = san_parts.copy()
                pseudo_san_parts.to_square = ""
                _append_pseudo_san(san, pseudo_san_parts, and_without_capture=False)

            # Test if the destination square has only one piece that can capture onto it
            conditions = [
                lambda m: board.is_capture(m),
                lambda m: m.to_square == move.to_square,
            ]
            if len(moves_where_all_conditions_hold(legal_moves, conditions)) == 1:
                # This is the only capture that can be made onto this square
                # Ex. "xc6"
                pseudo_san_parts = san_parts.copy()
                pseudo_san_parts.piece = ""
                pseudo_san_parts.file = ""
                pseudo_san_parts.rank = ""
                _append_pseudo_san(san, pseudo_san_parts, and_without_capture=False)

            if capturing_piece_type == chess.PAWN:
                # Test if only one pawn can capture between these two files
                conditions = [
                    lambda m: board.is_capture(m),
                    lambda m: board.piece_type_at(m.from_square) == chess.PAWN,
                    lambda m: chess.square_file(m.from_square)
                    == chess.square_file(move.from_square),
                    lambda m: chess.square_file(m.to_square)
                    == chess.square_file(move.to_square),
                ]
                if len(moves_where_all_conditions_hold(legal_moves, conditions)) == 1:
                    # This is the only pawn that can capture in this file
                    # Ex. "bxc", "bc"
                    pseudo_san_parts = san_parts.copy()
                    pseudo_san_parts.piece = ""
                    pseudo_san_parts.rank = ""
                    pseudo_san_parts.to_square = chess.FILE_NAMES[
                        chess.square_file(move.to_square)
                    ]
                    _append_pseudo_san(san, pseudo_san_parts, and_without_capture=True)

                # Test if this is the only way this pawn can capture
                conditions = [
                    lambda m: board.is_capture(m),
                    lambda m: m.from_square == move.from_square,
                ]
                if len(moves_where_all_conditions_hold(legal_moves, conditions)) == 1:
                    # This is the only way this pawn can capture
                    # Ex. "bx"
                    pseudo_san_parts = san_parts.copy()
                    pseudo_san_parts.to_square = ""
                    _append_pseudo_san(san, pseudo_san_parts, and_without_capture=False)

        # TODO we can do a lot more here!

    # Sanity check for duplicates. If there's an error here,
    # something about the logic above is wrong.
    all_pseudo_sans = list(chain.from_iterable(sans.values()))
    test_all_pseudo_sans = set()
    for pseudo_san in all_pseudo_sans:
        if pseudo_san in test_all_pseudo_sans:
            assert False, f"Internal error: duplicate pseudo-SAN: {pseudo_san}"
        test_all_pseudo_sans.add(pseudo_san)

    return dict(sans)


def get_pseudo_sans_to_sans_map(board: Board) -> dict[str, str]:
    sans_to_pseudo_sans_map = get_sans_to_pseudo_sans_map(board)
    return {ps: s for s, pss in sans_to_pseudo_sans_map.items() for ps in pss}


"""Bbox isolation functions"""


def get_scoresheet_bbox(image: Image) -> YoloBbox:
    raise NotImplementedError("TODO")


def get_headers_section_bboxes(scoresheet_image: Image) -> list[YoloBbox]:
    raise NotImplementedError("TODO")


def get_moves_section_bboxes(scoresheet_image: Image) -> list[YoloBbox]:
    raise NotImplementedError("TODO")


def get_column_bboxes(moves_section_image: Image) -> list[YoloBbox]:
    raise NotImplementedError("TODO")


def get_move_cell_bboxes(moves_section_image: Image) -> list[YoloBbox]:
    raise NotImplementedError("TODO")


def get_handwriting_blob_bboxes(section_image: Image) -> list[YoloBbox]:
    raise NotImplementedError("TODO")


def get_handwriting_char_bboxes(blob_image: Image) -> list[YoloBbox]:
    raise NotImplementedError("TODO")


def get_move_cell_bboxes_by_ply(moves_section_image: Image) -> dict[int, YoloBbox]:
    move_cell_bboxes = get_move_cell_bboxes(moves_section_image)
    column_bboxes = get_column_bboxes(moves_section_image)

    move_cell_bboxes_by_ply = {}
    for move_cell_bbox in move_cell_bboxes:
        ...


def get_char_probs(image: Image) -> dict[str, float]:
    """
    Use a classifier NN to get a vector of probabilities over the possible
    SAN characters for the given image, which contains a single handwritten
    character. Return a dict mapping each character to its probability,
    sorted in descending order.

    Supported characters should be: abcdefgh12345678xNBRQKO-+#=P!?
    """
    raise NotImplementedError("TODO")


def get_top_k_most_likely_sans(
    board: Board,
    handwriting_blob_image: Image,
    k: int = 1,
    beam_width: int = 100,
    excluded_sans: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """
    Given a board state and an image containing a handwritten move, return a list of `k`
    3-tuples `(pred_str, san, pred_loss)`:
      - `pred_str`: The predicted string, either a valid SAN or a pseudo-SAN
      - `san`: A valid SAN string corresponding to the written move
      - `pred_loss`: The mean squared error over the prediction probabilities of each char in `pred_str`.
        Lower values are better.
    The returned list is sorted by `pred_loss` (lowest/best values first).

    This function does a beam search over the legal SAN/pseudo-SAN moves to find the top `k`
    with the lowest loss. The `beam_width` defaults to `k * 10` if left as `None`.
    """
    if excluded_sans is None:
        excluded_sans = []
    if beam_width is None:
        beam_width = k * 10

    def _update_running_predicion_with_new_char_pred(
        pred_str: str,
        pred_loss: float,
        possible_pseudo_sans: set[str],
        pred_char: str,
        pred_char_prob: float,
    ) -> tuple[str, float, set[str]]:
        """
        Return a 3-tuple `(new_pred_str, new_pred_loss, new_possible_pseudo_sans)`:
        - `new_pred_str`: The current prediction characters as a string
        - `new_pred_loss`: The MSE over all predicted chars in `pred_str`
        - `new_possible_pseudo_sans`: Any SANs in `possible_pseudo_sans` that start with `new_pred_str`
        """
        new_pred_str = pred_str + pred_char
        new_pred_loss = ((pred_loss * len(pred_str)) + (1-pred_char_prob) ** 2) / len(new_pred_str)
        new_possible_pseudo_sans = {ps for ps in possible_pseudo_sans if ps.startswith(new_pred_str)}

        return new_pred_str, new_pred_loss, new_possible_pseudo_sans

    def _sort_beam_candidates(candidates: tuple[str, float, set[str]]) -> list[tuple[str, list[float], set[str]]]:
        """Sort the given `candidates` on their losses, low to high (best to worst). Returns a new `set`."""
        return sorted(candidates, key=lambda c: c[1], reverse=False)

    def _select_beam_candidates(candidates: list[tuple[str, float, set[str]]]) -> list[tuple[str, list[float], set[str]]]:
        """
        Sort the given `candidates` using `_beam_candidate_loss()` and select
        the top `beam_width` candidates and returns the values for the new beam.
        """
        return _sort_beam_candidates(candidates)[:beam_width]        

    pseudo_sans_to_sans_map = get_pseudo_sans_to_sans_map(board)
    """A mapping of possible pseudo-SANs to their corresponding legal move in correct SAN notation."""
    initial_pseudo_sans: set[str] = set(pseudo_sans_to_sans_map.keys())
    """A `set` of all pseudo-SANs that could represent a legal move in this position."""

    beam: list[tuple[str, float, set[str]]] = []
    """
    A list of `(pred_str, pred_loss, possible_pseudo_sans)` tuples:
    - `pred_str`: strings with a current running prediction over the already-seen chars
    - `pred_loss`: the MSE over the prediction probabilities for each char in `pred_str`
    - `possible_pseudo_sans`: A set of all legal pseudo-SANs that start with `pred_str`
    """

    char_bboxes = get_handwriting_char_bboxes(handwriting_blob_image)
    for char_bbox in char_bboxes:
        # Get a probability map over possible characters
        char_image = crop_to_bbox(handwriting_blob_image, char_bbox)
        char_probs = get_char_probs(char_image)

        # Now that we have the probability map, we want to get ALL
        # next possible candidates for the new beam, then take the
        # top-`beam_width` of them as the new beam. This means we
        # might have lots more than `beam_width` elements in `beam_candidates`
        # (which is fine, we throw them out on each iter).
        # See `beam` docs for format of the inner tuples.
        beam_candidates: list[tuple[str, list[float], set[str]]] = []

        # Initialize the empty beam on the first iter
        if not beam:
            for pred_str, pred_prob in char_probs.items():
                possible_pseudo_sans = {ps for ps in initial_pseudo_sans if ps.startswith(pred_str)}
                if not possible_pseudo_sans:
                    # There are no legal SANs or possible pseudo-SANs
                    # that start with this character
                    continue

                beam_candidates.append(pred_str, [pred_prob])
            beam = _select_beam_candidates(beam_candidates)
            continue

        # On subsequent iters, get all new beam candidates and select the top `beam_width` ones after
        for running_prediction in beam:
            # Unpack the current beam element
            pred_str, pred_loss, possible_pseudo_sans = running_prediction

            # Add a candidate for each possible `pred_char`, if any pseudo-SAN starts with
            # the string built by appending the `pred_char` to the current beam element's `pred_str`
            for pred_char, pred_char_prob in char_probs.items():
                new_running_prediction = _, _, new_possible_pseudo_sans = (
                    _update_running_predicion_with_new_char_pred(
                        pred_str,
                        pred_loss,
                        possible_pseudo_sans,
                        pred_char,
                        pred_char_prob
                    )
                )

                if not new_possible_pseudo_sans:
                    # There are no legal SANs or possible pseudo-SANs
                    # that start with this string
                    continue

                beam_candidates.append(new_running_prediction)

        # Set the new beam to the top `beam_width` candidates
        beam = _select_beam_candidates(beam_candidates)

    # Done with the beam search, just get the top `k` from the beam.
    # Remove candidates from the beam that don't have exactly 1 pseudo-SAN left
    # (these are predictions that would need more chars after those already
    # predicted to be a legal pseudo-SAN)
    # TODO: Investigate which predictions are getting removed here
    #   and if they could be valuable to keep
    beam = [r for r in _sort_beam_candidates(beam) if len(r[2]) == 1]

    results: list[tuple[str, str, float]] = []
    for prediction in beam[:k]:
        pred_str, pred_loss, _ = prediction
        assert pred_str in pseudo_sans_to_sans_map, \
            f'Internal error: `{pred_str = }` should be in `pseudo_sans_to_sans_map` here\n{pseudo_sans_to_sans_map = }'
        results.append((pred_str, pseudo_sans_to_sans_map[pred_str], pred_loss))

    return results


def scoresheet_2_pgn(image: Image) -> str:
    scoresheet_bbox = get_scoresheet_bbox(image)
    scoresheet_image = crop_to_bbox(image, scoresheet_bbox)

    headers_section_bboxes = get_headers_section_bboxes(scoresheet_image)

    board = Board()

    moves_section_bboxes = get_moves_section_bboxes(scoresheet_image)
    for moves_section_bbox in moves_section_bboxes:
        moves_section_image = crop_to_bbox(scoresheet_image, moves_section_bbox)

        handwriting_blob_bboxes = get_handwriting_blob_bboxes(moves_section_image)
        for handwriting_blob_bbox in handwriting_blob_bboxes:
            handwriting_blob_image = crop_to_bbox(
                moves_section_image, handwriting_blob_bbox
            )

            # top_k_most_likely_sans = get_top_k_most_likely_sans(
            #     board,
            #     handwriting_blob_image,
            #     # TODO
            # )
