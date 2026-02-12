## Single GPU speedrun

I tried estimating the percentage of people who can afford to fool around on an 8xH100 instance. I did the math in `float16` and got `0.0`.

I wanted to set up a single-GPU speedrun where the emphasis is on accessibility, learning, and fun--even if it means compromising some on the relevance of the results and the robustness of the leaderboard.

A major challenge with running on a single GPU--and why 8xH100s make a lot of sense--is that the result of a single run is much less informative than we would all like to believe. For modded-nanogpt, my default is to run anything I try _4 times_ before even looking at the results. Once you think you have something good, you'll typically need to do 6-10 runs to prepare your submission, because you need to prove with 99% confidence that your model hits the required validation loss. And quite often you don't quite make it!

You can do all of that on a single gpu, it's just going to take a lot of time and patience and still be fairly expensive. 

I've thought a lot about how to make this work, and here's my current thinking on the structure.

**Baselines**

Getting to set a record in a challenging speedrun competition is really motivating and rewarding. But running a competition like that requires a lot of effort to ensure fairness. I'd really like for this project to be easier to maintain, and feel more collaborative.

My thought is, instead of accepting carefully validated records, we'll operate in terms of "baselines". 

We'll create a new baseline when a target is reached--e.g., we've reduced the time by 10%.

Within a baseline, things are pretty loose. Lucky runs are fine. i.e., if you try an idea and get a good result, you can just submit that, no need to prove significance. We'll sort that out more collectively.

Anyone can contribute even if it's just by re-running existing code and getting a better result.

When it's time for a new baseline, we'll try and credit everyone involved and note what they contributed.

**Submission Format**

Submissions are handled by providing links to wandb runs rather than adding code or log files to the repo (those will be attached to the wandb run instead).

I want to be able to accept runs without doing thorough verification--it will have to be more of a community effort to catch mistakes.

I'm not sure of the best way to manage this. To start out, I think we could just drop them in a discussion thread, and I'll maintain it manually until everything is more established?

**Pull Requests**

PRs are still welcome for code changes, whether quality of life, or to fold improvements into the current baseline source file.

PRs with performance improvements should still be backed by the necessary run data and statistics, but it doesn't need to be one person who contributes the idea, performs the requisite runs, and assembles the PR. We'll credit everyone involved.

**GPUs**

Run on whatever! Run the baseline and your code on your hardware and share the results. We'll accept scores from any hardware.

For the actual "fastest time", I'm thinking we'll have an Ampere track and a Hopper track, and presumably those records will be set by running on an A100 or an H100, respectively. 

