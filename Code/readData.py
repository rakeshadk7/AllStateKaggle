import pandas as pd



train = pd.read_csv("train.csv")
_columns = list(train)

_cat = [x for x in _columns if x.startswith("cat")]
_con = [x for x in _columns if x.startswith("con")]


# This loops tries to encode the values for CAT variables.
#Takes too long.can you please try and make it efficient when have time .

for cat_column_name in _cat:

        uniq_col_values = list(set(train[cat_column_name]))

        for uniq_value in uniq_col_values:
                train.loc[train[cat_column_name] == uniq_value, "{}_{}".format(
                    cat_column_name, uniq_value)] = 1
                train.loc[train[cat_column_name] != uniq_value, "{}_{}".format(
                    cat_column_name, uniq_value)] = 0


# Trying to create sparse matrix so that we can feed the data into XGBoost
