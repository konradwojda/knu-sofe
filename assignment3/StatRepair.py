import inspect
from typing import Any, Optional

from StatisticalDebugger import ContinuousSpectrumDebugger, RankingDebugger


class StatRepair(ContinuousSpectrumDebugger, RankingDebugger):
    def suspiciousness(self, event: Any) -> Optional[float]:
        failed = len(self.collectors_with_event(event, self.FAIL))
        not_in_failed = len(self.collectors_without_event(event, self.FAIL))
        not_in_passed = len(self.collectors_without_event(event, self.PASS))
        passed = len(self.collectors_with_event(event, self.PASS))

        try:
            # return failed / math.sqrt((failed + not_in_failed) * (failed + passed))
            return 2 * (failed + (not_in_passed / (passed + not_in_passed)))
        except ZeroDivisionError:
            return None
        
    def mostsimilarstmt(self, targetloc):
        functions = self.covered_functions()
        cov = self.coverage()
        found_line = None
        closest_line = None
        dist = None
        for function in functions:
            if function.__name__ == targetloc[0]:
                try:
                    source_lines, starting_line_number = inspect.getsourcelines(function)
                except OSError:
                    continue
                line_number = starting_line_number

                for line in source_lines:
                    if line_number == targetloc[1]:
                        found_line = line
                    line_number += 1

        found_line = found_line.strip()

        for function in functions:
            try:
                source_lines, starting_line_number = inspect.getsourcelines(function)
            except OSError:
                continue

            line_no = starting_line_number
            for line in source_lines:
                if(function, line_no) in cov:
                    stripped = line.strip()
                    distance = levenshteinDistance(stripped, found_line)
                    if not dist:
                        dist = distance
                    if distance != 0 and distance < dist:
                        dist = distance
                        closest_line = stripped
                line_no += 1


        return (closest_line, dist)

def levenshteinDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]