import pandas as pd
import io

# 假设 CSV 数据为一个字符串，可以替换为从文件中读取的方式
data = """1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
2|GoldenEye (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?GoldenEye%20(1995)|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0
3|Four Rooms (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Four%20Rooms%20(1995)|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0"""
df = pd.read_table('../ml-100k/u.item', sep='|', header=None)
# 使用 StringIO 来模拟文件对象
# 提取第二列（电影名称）
movie_titles = df[1].tolist()

# 打印电影名称
for title in movie_titles:
    print(title)
