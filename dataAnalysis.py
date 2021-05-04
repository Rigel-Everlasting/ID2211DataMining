import gzip
import json

with gzip.open('./models/dataset/Digital_Music_5.json.gz', 'rb') as g:
    data = {}
    i = 0

    fiveStar = 0
    fourStar = 0
    threeStar = 0
    twoStar = 0
    oneStar = 0
    items = set()
    users = set()

    for line in g:
        line = json.loads(line)
        keys = ['reviewerID', 'asin', 'overall', 'unixReviewTime']
        line['overall'] = int(line['overall'])
        l = {k: line[k] for k in keys}
        data[i] = l
        
        # TODO: Find the number of unique users
        users.add(data[i]["reviewerID"])

        # TODO: Find the number of unique items(asin)
        items.add(data[i]["asin"])

        # TODO: Find the number of stars
        if data[i]["overall"] == 5:
            fiveStar += 1

        if data[i]["overall"] == 4:
            fourStar += 1

        if data[i]["overall"] == 3:
            threeStar += 1

        if data[i]["overall"] == 2:
            twoStar += 1

        if data[i]["overall"] == 1:
            oneStar += 1
        i += 1

print("Total length of data %d" %len(data))
print("Total five star ratings %d" %fiveStar)
print("Total four star ratings %d" %fourStar)
print("Total three star ratings %d" %threeStar)
print("Total two star ratings %d" %twoStar)
print("Total one star ratings %d" %oneStar)
print("Total number of unique users %d" %len(users))
print("Total number of unique items %d" %len(items))

