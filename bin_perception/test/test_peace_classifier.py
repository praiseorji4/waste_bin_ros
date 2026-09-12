"""Unit tests for the peace-sign geometry, using synthetic hand landmarks.

No camera and no mediapipe needed: classify_peace() takes plain (x, y) pixel points,
so the gesture logic can be tested on its own.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bin_perception.peace_detector import (  # noqa: E402
    classify_peace, finger_extended, thumb_extended, v_angle_deg, FINGERS,
)


def make_hand(index_len=1.0, middle_len=1.0, ring_len=0.35, pinky_len=0.3,
              thumb_len=0.35, spread_deg=20.0, rotate_deg=0.0, scale=100.0,
              origin=(300.0, 400.0)):
    """Build 21 landmarks for a hand pointing 'up' (-y), then rotate by rotate_deg.

    Finger lengths are fractions of a fully extended finger: >0.6 reads as extended,
    ~0.3 reads as folded.
    """
    pts = [(0.0, 0.0)] * 21

    def place(idx, x, y):
        pts[idx] = (x, y)

    place(0, 0.0, 0.0)                      # wrist
    # MCP knuckles spread across the palm
    place(5, -0.25, -0.55)                  # index mcp
    place(9, -0.02, -0.60)                  # middle mcp
    place(13, 0.20, -0.57)                  # ring mcp
    place(17, 0.40, -0.50)                  # pinky mcp
    place(2, -0.45, -0.20)                  # thumb mcp

    def finger(mcp_i, pip_i, tip_i, length, angle_deg):
        """length ~1.0 = straight out; length < 0.5 = curled back into the palm."""
        mx, my = pts[mcp_i]
        a = math.radians(angle_deg)
        dx, dy = math.sin(a) * 0.75, -math.cos(a) * 0.75
        # the PIP knuckle barely moves whether the finger is open or closed
        px, py = mx + dx * 0.45, my + dy * 0.45
        place(pip_i, px, py)
        if length >= 0.6:
            place(tip_i, mx + dx * length, my + dy * length)
        else:
            # curled: the tip comes back towards the wrist (as in a fist)
            curl = 0.9 * (1.0 - length)
            place(tip_i, px + (0.0 - px) * curl, py + (0.0 - py) * curl)

    finger(5, 6, 8, index_len, -spread_deg / 2)
    finger(9, 10, 12, middle_len, spread_deg / 2)
    finger(13, 14, 16, ring_len, 12.0)
    finger(17, 18, 20, pinky_len, 22.0)
    # thumb: out to the side
    tx, ty = pts[2]
    place(3, tx - 0.25 * thumb_len, ty - 0.12 * thumb_len)
    place(4, tx - 0.55 * thumb_len, ty - 0.25 * thumb_len)

    r = math.radians(rotate_deg)
    cos_r, sin_r = math.cos(r), math.sin(r)
    out = []
    for x, y in pts:
        xr = x * cos_r - y * sin_r
        yr = x * sin_r + y * cos_r
        out.append((origin[0] + xr * scale, origin[1] + yr * scale))
    return out


def test_peace_sign_is_detected():
    ok, d = classify_peace(make_hand())
    assert ok, d
    assert d['index'] and d['middle']
    assert not d['ring'] and not d['pinky']


def test_open_palm_is_rejected():
    ok, d = classify_peace(make_hand(ring_len=1.0, pinky_len=1.0))
    assert not ok
    assert d['ring'] and d['pinky']


def test_fist_is_rejected():
    ok, _ = classify_peace(make_hand(index_len=0.3, middle_len=0.3))
    assert not ok


def test_pointing_one_finger_is_rejected():
    ok, _ = classify_peace(make_hand(middle_len=0.3))
    assert not ok


def test_two_fingers_held_together_is_rejected():
    """Index+middle extended but parallel is a 'salute', not a peace sign."""
    ok, d = classify_peace(make_hand(spread_deg=2.0))
    assert not ok
    assert d['v_angle'] < 12.0


def test_rotated_hands_still_detected():
    """Orientation independence: the same gesture at many angles."""
    for angle in (0, 45, 90, 135, 180, 225, 270, 315):
        ok, d = classify_peace(make_hand(rotate_deg=angle))
        assert ok, f'failed at {angle} deg: {d}'


def test_small_and_large_hands():
    for scale in (40.0, 100.0, 260.0):
        ok, _ = classify_peace(make_hand(scale=scale))
        assert ok, f'failed at scale {scale}'


def test_v_angle_matches_spread():
    for spread in (20.0, 40.0):
        d = v_angle_deg(make_hand(spread_deg=spread))
        assert abs(d - spread) < 6.0, f'spread {spread} -> measured {d}'


def test_wide_splayed_fingers_rejected():
    ok, d = classify_peace(make_hand(spread_deg=95.0))
    assert not ok
    assert d['v_angle'] > 70.0


def test_thumb_flag():
    pts = make_hand(thumb_len=1.0)
    assert thumb_extended(pts)
    assert classify_peace(pts, allow_thumb=True)[0]
    assert not classify_peace(pts, allow_thumb=False)[0]


def test_finger_extended_helper():
    pts = make_hand()
    assert finger_extended(pts, *FINGERS['index'])
    assert not finger_extended(pts, *FINGERS['pinky'])


if __name__ == '__main__':
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS', name)
                passed += 1
            except AssertionError as exc:
                print('FAIL', name, '->', exc)
                failed += 1
    print(f'\n{passed} passed, {failed} failed')
    sys.exit(1 if failed else 0)
