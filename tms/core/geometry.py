def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def union(boxes):
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def link_riders(persons, bikes, min_overlap=0.3):
    links = {}
    for person_id, box in persons.items():
        lower = (box[0], (box[1] + box[3]) // 2, box[2], box[3])
        if not area(lower):
            continue
        best_id, best_score = None, min_overlap
        for bike_id, bike in bikes.items():
            score = intersection(lower, bike) / area(lower)
            if score >= best_score:
                best_id, best_score = bike_id, score
        if best_id is not None:
            links.setdefault(best_id, []).append(person_id)
    return links


def head_box(person):
    x1, y1, x2, y2 = person
    return (x1, y1, x2, y1 + round((y2 - y1) * 0.3))


def windscreen_box(car):
    x1, y1, x2, y2 = car
    width, height = x2 - x1, y2 - y1
    return (x1 + round(width * 0.05), y1 + round(height * 0.12),
            x1 + round(width * 0.58), y1 + round(height * 0.55))
