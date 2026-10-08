#pragma once

// llama_qsa_select_cells — the cells a QSA query attends: its top n_top whole blocks, then its incomplete tail
// (fork-kernel-fusion, invariant S1 in tests/INVARIANTS.md).
//
// The selection is made over blocks, never over cells: every cell of a block carries the block's score, so a
// cell-level top-k whose width is not a whole number of blocks cuts a block, and which of its tied cells survive
// is the order the top-k kernel happens to resolve ties in. Here a block is in or out whole, and the tail is
// named by the host rather than competing for slots.
//
//   score     F32 [n_blocks, n_tps, n_stream]  block score plus the host's selectability bias: 0 for a whole
//                                              block wholly before the query's tail, -inf for anything else
//   blk_cells I32 [r*n_blocks, n_stream]       member cells of each block, in position order
//   tail_idxs I32 [r - 1, n_tps, n_stream]     the query's tail cells, padded with the sentinel n_kv; nullptr when r == 1
//
// returns I32 [r*n_top + r - 1, n_tps, n_stream]. A block picked only to fill the budget (fewer selectable blocks than
// n_top) is -inf, so it is foreign, in the future or the tail block itself: the attention mask drops its cells or the
// tail already names them, and the attended set does not change. The sentinel n_kv needs a mask with one spare row.
#include "ggml.h"

static inline ggml_tensor * llama_qsa_select_cells(
        ggml_context * ctx,
        ggml_tensor  * score,
        ggml_tensor  * blk_cells,
        ggml_tensor  * tail_idxs,
        int64_t        r,
        int64_t        n_top) {
    const int64_t n_blocks = score->ne[0];
    const int64_t n_tps    = score->ne[1];
    const int64_t n_stream = score->ne[2];

    GGML_ASSERT(n_top > 0 && n_top <= n_blocks);
    GGML_ASSERT(blk_cells->ne[0] == r*n_blocks && blk_cells->ne[1] == n_stream);

    ggml_tensor * top = ggml_top_k(ctx, score, (int) n_top); // [n_top, n_tps, n_stream], unordered

    // get_rows gathers per stream: row s of the index list reads stream s's own block table
    ggml_tensor * table = ggml_reshape_3d(ctx, blk_cells, r, n_blocks, n_stream);
    ggml_tensor * cells = ggml_get_rows(ctx, table, ggml_reshape_2d(ctx, top, n_top*n_tps, n_stream));
    cells = ggml_reshape_3d(ctx, cells, r*n_top, n_tps, n_stream);

    if (r > 1) {
        GGML_ASSERT(tail_idxs != nullptr);
        GGML_ASSERT(tail_idxs->ne[0] == r - 1 && tail_idxs->ne[1] == n_tps && tail_idxs->ne[2] == n_stream);
        cells = ggml_concat(ctx, cells, tail_idxs, 0);
    }

    return cells;
}
