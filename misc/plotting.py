import numpy as np


def generate_diverse_colors(n_colors, n_global_moves=32, ref_color=None):
    # ref_color: a list of numpy array with size 3 (r,g,b) and values in [0, 1]
    
    class Color:
        max_weighted_square_distance = (((512 + 127) * 65025) >> 8) + 4 * 65025 + (((767 - 127) * 65025) >> 8)

        def __init__(self, r, g, b):
            self.r, self.g, self.b = r, g, b

        def weighted_square_distance(self, other):
            rm = (self.r + other.r) // 2  # integer division
            dr =  self.r - other.r
            dg =  self.g - other.g
            db =  self.b - other.b
            return (((512 + rm) * dr*dr) >> 8) + 4 * dg*dg + (((767 - rm) * db*db) >> 8)

        def min_weighted_square_distance(self, index, others):
            min_wsd = self.max_weighted_square_distance
            for i in range(0, len(others)):
                if i != index:
                    wsd = self.weighted_square_distance(others[i])
                    if  min_wsd > wsd:
                        min_wsd = wsd
            return min_wsd

        def is_valid(self):
            return 0 <= self.r <= 255 and 0 <= self.g <= 255 and 0 <= self.b <= 255

        def add(self, other):
            return Color(self.r + other.r, self.g + other.g, self.b + other.b)

        def __repr__(self):
            return f"({self.r}, {self.g}, {self.b})"

    colors_hex = [
        "000000", "00FF00", "0000FF", "FF0000", "01FFFE", "FFA6FE", "FFDB66", "006401",
        "010067", "95003A", "007DB5", "FF00F6", "FFEEE8", "774D00", "90FB92", "0076FF",
        "D5FF00", "FF937E", "6A826C", "FF029D", "FE8900", "7A4782", "7E2DD2", "85A900",
        "FF0056", "A42400", "00AE7E", "683D3B", "BDC6FF", "263400", "BDD393", "00B917",
        "9E008E", "001544", "C28C9F", "FF74A3", "01D0FF", "004754", "E56FFE", "788231",
        "0E4CA1", "91D0CB", "BE9970", "968AE8", "BB8800", "43002C", "DEFF74", "00FFC6",
        "FFE502", "620E00", "008F9C", "98FF52", "7544B1", "B500FF", "00FF78", "FF6E41",
        "005F39", "6B6882", "5FAD4E", "A75740", "A5FFD2", "FFB167", "009BFF", "E85EBE",
    ]
    colors = list(map(lambda h: Color(*tuple(int(h[i:i+2], 16) for i in (0, 2, 4))), colors_hex))
    
    if ref_color is not None:
        n_colors += 1
        
    if len(colors) >= n_colors:
        colors = colors[:n_colors]
        if ref_color is not None:
            ref_color = Color(int(ref_color[0] * 255), int(ref_color[1] * 255), int(ref_color[2] * 255))
            min_dist = float("inf")
            min_idx = 0
            for idx, c in enumerate(colors):
                dist = c.weighted_square_distance(ref_color)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            colors.pop(min_idx)
    else:
        colors = [Color(127, 127, 127) for i in range(0, n_colors)]
        if ref_color is not None:
            colors[0] = Color(int(ref_color[0] * 255), int(ref_color[1] * 255), int(ref_color[2] * 255))
            starting_idx = 1
        else:
            starting_idx = 0

        steps = [Color(dr, dg, db) for dr in [-1, 0, 1]
                                   for dg in [-1, 0, 1]
                                   for db in [-1, 0, 1] if dr or dg or db]  # i.e., except 0,0,0
        moved = True
        global_move_phase = False
        global_move_count = 0
        while moved or global_move_phase:
            moved = False
            for index in range(starting_idx, len(colors)):
                color = colors[index]
                if global_move_phase:
                    best_min_wsd = -1
                else:
                    best_min_wsd = color.min_weighted_square_distance(index, colors)
                for step in steps:
                    new_color = color.add(step)
                    if new_color.is_valid():
                        new_min_wsd = new_color.min_weighted_square_distance(index, colors)
                        if  best_min_wsd < new_min_wsd:
                            best_min_wsd = new_min_wsd
                            colors[index] = new_color
                            moved = True
            if not moved:
                if  global_move_count < n_global_moves:
                    global_move_count += 1
                    global_move_phase = True
            else:
                global_move_phase = False

        if ref_color is not None:
            colors = colors[1:]
    colors_np = np.array([[c.r, c.g, c.b] for c in colors]) / 255
    return colors_np


def legend_without_duplicate_labels(ax):
    # Source: https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    handles, labels = zip(*unique)
    return handles, labels