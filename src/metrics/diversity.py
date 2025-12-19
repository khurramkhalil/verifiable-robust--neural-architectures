from Levenshtein import distance

def population_diversity(arch_strings):
    """
    Computes average pairwise edit distance between architecture strings.
    """
    if len(arch_strings) < 2:
        return 0.0
    
    total_dist = 0
    count = 0
    for i in range(len(arch_strings)):
        for j in range(i + 1, len(arch_strings)):
            total_dist += distance(arch_strings[i], arch_strings[j])
            count += 1
            
    return total_dist / count if count > 0 else 0.0
