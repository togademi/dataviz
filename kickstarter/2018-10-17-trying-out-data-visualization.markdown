# Trying out data visualization

This past summer I wanted to improve my understanding of machine learning by doing some projects and following tutorials. I quickly realized that I was only focusing on the application of machine learning techniques, but I wasn't really understanding how to manipulate the data (organizing the data is what you do first, before applying any ML algorithm to it). Thus in order to learn how to handle and process data, I decided to make some **data visualization**.

I browsed the [datasets available on Kaggle](https://www.kaggle.com/datasets) until I found one that caught my attention: [Kickstarter projects from 2009 to 2017](https://www.kaggle.com/kemical/kickstarter-projects).

![Kickstarter logo](./img/kickstarter-logo-color.png)

I observed the different columns in the dataset and decided that I wanted to visualize the success rate of Kickstarter projects by categories (like *photography*, *games*, *art*, *design*, etc). Then I proceeded to organize the data to fit this interest.

*Disclaimer*: here when I talk about successful projects I mean projects that attained their funding goal. I know that there are some projects which achieved their goals but never delivered what was promised afterward, but I don't have an easy way to know which ones by looking at the data set.

## Organizing data

I used Python for this. Firstly the **Pandas library** (and sometimes NumPy too) to create a data frame with only the data I was interested in.

So I extracted the different Kickstarter categories from the original data set (called 'data' here):

```
main_categories = data["main_category"].values
main_categories = np.unique(main_categories)
main_categories = pd.Series(data=main_categories)
```

And then split the *games* category into three categories: *video games*, *tabletop games* and *other games*. I did this because I was told that tabletop games have a much higher success rate than video games on Kickstarter, so it would be wise to have insights on each of them.

```
games = ["Tabletop Games", "Video Games"]
games = pd.Series(data=games)

categories = pd.concat([main_categories, games], ignore_index=True)
df_minus_video_tabletop = data[(data.category != "Tabletop Games") & (data.category != "Video Games")]
```

Then I went to count, for each category, **how many projects had attained their goal, undershot it or been canceled**. I also counted the **funding goal** (in USD) for each category. I made two functions to do this, that they would go inside a loop over every category.

```
def items_in_category(df, category, state, is_main_category):
    if is_main_category:
        temp = df[(df.main_category == category) & (df.state == state)]['state'].value_counts()
    else:
        temp = df[(df.category == category) & (df.state == state)]['state'].value_counts()
    return temp.at[state]

def goal_sum(df, category, is_main_category):
    if is_main_category:
        return df[df.main_category == category]["usd_goal_real"].sum()
    else:
        return df[df.category == category]["usd_goal_real"].sum()
```

And to finish organizing data I needed to:

* count the **total number of projects** in each category (to calculate the success rate)
* calculate the **success rate**
* calculate the **average funding goal** (I'll explain later why)

```
# Create new column for total number of projects
categories_df['total_projects'] = categories_df.apply(lambda row: row.attained_goal + row.undershot_goal + row.canceled,
                                                      axis=1)

# Create new column for success ratio
categories_df['success_ratio'] = categories_df.apply(lambda row: (row.attained_goal) / (row.undershot_goal + row.canceled),
                                                     axis=1)

# Create new column for average goal
categories_df['average_goal'] = categories_df.apply(lambda row: row.goal_sum / row.total_projects,
                                                     axis=1)
```

After doing this and **sorting** the different categories **by success rate**, we get this data frame:

![Data frame of the Kickstarter categories](./img/kickstarter_df.jpg)


## Plotting data

Once we have the data frame with all the data that interests us, it's time to plot for visualizing it. To plot all this data I chose a **stacked bar graph that shows the relative success/failure/cancelation rates** among categories. I used the **Matplotlib library** for this.

```
fig, ax = plt.subplots()

attained_goal_ax = ax.bar(x, attained_goal, width, color='#034752', edgecolor='white')
undershot_goal_ax = ax.bar(x, undershot_goal, width, bottom=attained_goal, color='#0c9ab2', edgecolor='white')
canceled_ax = ax.bar(x, canceled, width, bottom=[i+j for i,j in zip(attained_goal, undershot_goal)],
                     color='#00daff', edgecolor='white')
```

Then I wanted to **show the average funding goal** of each category. To do this I thought of placing a graphical indication (like a small symbol) in each of the bars at the height of their corresponding *average funding goal* (you see this on the Y-axis on the right). What I did was plotting a **line graph** over the existing graph. Into this line plot I put markers in the shape of a stripe (`marker='_'`), then I removed the line (`linestyle='None'`), **leaving only the markers**.

```
ax2 = ax.twinx()
average_goal = ax2.plot(x, categories_df["average_goal"], linestyle='None', color="white",
                        marker='_', markersize=39, markeredgewidth=7, markeredgecolor='#e58200')
plt.ylabel("Average funding goal (USD)", fontsize=19, color='#e58200')
```

***Et voilà!***

![Graph: success rate of Kickstarter projects by category](./img/ks_success_ratio_avg_goal.jpg)

Why I wanted to calculate the average funding goal was to compare it with the success rate. Therefore we see that the categories that most often achieve their goals are also some of the ones that ask for the least money. This seems reasonable. Let's also look at the two outliers here:

* the ***film and video*** category is the 7th most **successful**, while also being the 3rd category that **asks for the most money**.
* the ***crafts*** category is one of the **least successful** despite being the 2nd category that **asks for the least money**.

This graph has actually taken me many iterations and reconsiderations because every time that I asked for feedback on the internet I got many replies telling me how to improve it. [This was my first version.](./img/ks_success_ratio_first_version.jpg)

---

You can find the Jupyter notebook with all the code [here](https://github.com/togademi/dataviz/blob/master/kickstarter/kickstarter.ipynb).

And the [original dataset](https://www.kaggle.com/kemical/kickstarter-projects).
