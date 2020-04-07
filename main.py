from DLC_analysis_class import *

test = DLC_analysis(path = '../../Desktop/DLC_social_1/')
print(test)

track_dict = test.load_tables()

print(track_dict)