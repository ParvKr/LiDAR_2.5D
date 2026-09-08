import numpy as np
r = np.array([-1, 0, 1]) + 20000
c = np.array([-2, 0, 2]) + 20000
b = np.array([0, 1, 2])

packed = (r.astype(np.int64) << 32) | (c.astype(np.int64) << 16) | b.astype(np.int64)

for u_key in packed:
    b_out = int(u_key & 0xFFFF)
    c_out = int((u_key >> 16) & 0xFFFF) - 20000
    r_out = int((u_key >> 32) & 0xFFFFFFFF) - 20000
    print(r_out, c_out, b_out)
