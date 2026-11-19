# Updating attention state after a delayed insertion

I tested whether a small sample of unchanged tokens predicts how inserting an earlier dialogue changes the attention state of later tokens. After the insertion, the model rereads the first 32 later tokens. I pair their previous and current key and value vectors, remove the positional rotation from keys, and estimate the change separately for each attention head and cache type. Values are already in a common coordinate system.

Let $d_j$ be the vector after insertion minus the vector before insertion at anchor position $j$, where $j$ runs from 1 to 32. Let $a$ be the mean of the first 16 differences and $b$ the mean of the last 16. I estimate the fraction of the displacement that persists for another 16 positions as

$$
r=\min\!\left(0.99,\max\!\left(0,\frac{a^\top b}{a^\top a+10^{-8}}\right)\right).
$$

The superscript $\top$ denotes transposition. At the $i$-th skipped position, counting from zero immediately after the final anchor, I add $r^{(i+8.5)/16} b$ to the old cached vector. The offset of 8.5 measures the distance from the center of the last 16 anchors. For a key, I first rotate the displacement to the token's new position and add it to its repositioned old key. For a value, I add the displacement directly. I then replay the final 48 tokens on this corrected cache. The matched partial-replay control omits the displacement. Both use the old recurrent matrices at the end of the skipped region and perform the same terminal replay.

The fixed active suffix contains the first and last 1,024 tokens of the later transcript. The service keeps the complete incoming session and searches all dated turns available at the time of the question. The initial model cache contains up to 128 preceding tokens. Thus the text index and active model cache have different history lengths. Each comparison receives the same updated text and retrieved passages.

The implementation inverts rotary position in FP32 by reflecting the second coordinate of each rotary pair before and after a positive-position rotation. It preserves channels outside the rotary subspace. The check exercises one-token and 32-token windows at four starting positions. [Implementation](cache_rotary.py) · [Verification](check_rotary.py)
