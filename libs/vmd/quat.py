import math

_NLERP_DOT_THRESHOLD = 0.9995


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(c / n for c in q)


def conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def angle_deg(a, b):
    r = multiply(normalize(b), conjugate(normalize(a)))
    v = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    return math.degrees(2.0 * math.atan2(v, abs(r[3])))


def slerp(q0, q1, t):
    a = normalize([float(c) for c in q0])
    b = normalize([float(c) for c in q1])
    d = dot(a, b)
    if d < 0.0:
        b = tuple(-c for c in b)
        d = -d
    if d > _NLERP_DOT_THRESHOLD:
        return normalize(tuple(x + t * (y - x) for x, y in zip(a, b, strict=True)))
    diff = math.sqrt(sum((y - x) ** 2 for x, y in zip(a, b, strict=True)))
    total = math.sqrt(sum((y + x) ** 2 for x, y in zip(a, b, strict=True)))
    th0 = 2.0 * math.atan2(diff, total)
    th = th0 * t
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return tuple(s0 * x + s1 * y for x, y in zip(a, b, strict=True))
