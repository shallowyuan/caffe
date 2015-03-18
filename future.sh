#! /bin/bash

# Workflow:
# 
# 1. Run `./future.sh`
# 2. If you see conflicts, fix them and commit, then run `./future.sh`.
# 3. Repeat 2, until you see the message done in the result of `./future.sh`.

BASE_BRANCH="origin/master"
BRANCHES="origin/accum-grad origin/share-blobs-over-nets origin/pythonlayer-parameter origin/python-gpu-full origin/future-misc origin/python-gradient-checker-accum"
STATEFILE=".futurestate"

git status | grep "^You have unmerged paths.$" > /dev/null && { echo "**Fix conflicts and commit to conclude merge, then run this command again.**"; git ls-files -u | sed "s/\s/ /g" | cut -d" " -f4| uniq | sed "s/^/  /g" ; exit 1 ; }

git status | grep "^All conflicts fixed but you are still merging.$" > /dev/null && { echo "**Commit to conclude merge, then run this command again.**" ; exit 1 ; }

if ! [ -e $STATEFILE ]; then
    echo "Fetching all remote branches..."
    git fetch --all || exit 1
    echo "Resetting to $BASE_BRANCH..."
    git reset --hard $BASE_BRANCH || exit 1
fi
touch $STATEFILE

for branch in $BRANCHES
do
    grep "^running:$branch$" $STATEFILE > /dev/null
    if [ "$?" = 0 ] ; then
	echo "$branch already applied. skip..."
	continue
    fi
    echo "running:$branch" >> $STATEFILE
    echo "Merging $branch..."
    git merge --no-ff $branch -m "Merge $branch" || { echo "**Fix conflicts and commit to conclude merge, then run this command again.**" ; exit 1 ; }
done
echo "Done."
rm $STATEFILE
echo "**NOTE: Do NOT push this to origin/master. To revert merges, run \`git reset ---hard origin/master\`.**"
