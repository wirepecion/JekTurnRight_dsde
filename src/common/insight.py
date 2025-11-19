import pandas as pd

def Co_occurrence_analysis(df, correlation_check__tag):
    s = df['type'].value_counts()

    s_roads = s[s.index.str.contains(correlation_check__tag)]
    co_occurrence_counts = {}

    for tag_set_str, count in s_roads.items():
    
        tags = tag_set_str.strip('{}').split(',')
        if len(tags) == 1:
            co_occurrence_counts[correlation_check__tag] = co_occurrence_counts.get(correlation_check__tag, 0) + count
            
        else:
            for tag in tags:
                if tag != correlation_check__tag:
                    co_occurrence_counts[tag] = co_occurrence_counts.get(tag, 0) + count

    correlation_series = pd.Series(co_occurrence_counts).sort_values(ascending=False)
    return correlation_series