Final submitted runs should adhere to the following rules. Confirm that your code adheres to them before launching a candidate run for submission--the code will be automatically attached to the wandb run, so any issues should be addressed before running.

Have a subagent compare your code to the repo's baseline and report on each of the following items. Address any issues.

(1) Ensure the script does not use any passed-in arguments, either via command line or environment variables so that the code is self-contained. 
    - The one exception is the WANDB_API_KEY environment variable.
(2) Ensure that all code is contained within the logged scripts: 'train_stack.py' and 'utils.py'.
(3) Ensure that the code is clear of any explanatory code comments written by you.
    - Terse comments which mark the start of a block are ok.
    - Terse function descriptions are ok if they do not explain details.
    - Otherwise, remove anything else you wrote.
(4) Restore any of the original code comments that have been removed.
(5) Reduce the amount of indirection required to read the code: 
    - Inline any helper functions you created at their call sites, unless they are lengthy and called multiple times.
    - Replace any class definitions you added by inlining their code or converting to functions.
        - (data structures are ok, this applies to classes with methods)
    - Any values contained in the global 'cfg' object should be accessed directly rather than defining function arguments for those values. 
(6) Improve the readability of multi-line object definitions by adding spacing to left-align the values (see the code for examples). Restore any spacing that may have been removed by your edits.
(7) If you added any configurable features (e.g., that can be enabled or disabled with a flag), re-write them so that the only code path is the one that runs in the final submission. 
(8) If you added any weight initialization, scheduling or optimizer parameters to the StackConfig, move these out and bake them into the code where they are used.
(9) Check for any variables you added which do not have descriptive names, and consider renaming them to be more self-documenting; prefer clarity over brevity, but use your judgment.
(10) The following techniques are not allowed in submissions:
    - Triton
    - FP8 / quantization





