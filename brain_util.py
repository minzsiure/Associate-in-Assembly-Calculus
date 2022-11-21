def overlap(a, b):
    return len(set(a) & set(b))


def get_overlaps(winners_list, base, percentage=False):
    overlaps = []
    base_winners = winners_list[base]
    k = len(base_winners)
    for i in range(len(winners_list)):
        o = overlap(winners_list[i], base_winners)
        if percentage:
            overlaps.append(float(o)/float(k))
        else:
            overlaps.append(o)
    return overlaps
