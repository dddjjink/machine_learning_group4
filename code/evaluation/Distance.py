import Evaluation


# 距离公式
class Distance(Evaluation):
    def __str__(self):
        return 'Distance'

    def __call__(self, vec1, vec2, p):
        return self.minkowski_distance(vec1, vec2, p)

    # 闵可夫斯基距离
    def minkowski_distance(self, vec1, vec2, p):
        i = 0
        sum = 0
        while i < len(vec1):
            sum += abs(vec1[i] - vec2[i]) ** p
            i += 1
        if p == 1:
            return sum
        elif p == 2:
            return sum ** 0.5
