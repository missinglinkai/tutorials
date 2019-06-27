# Setup MissingLink Code Tracking

## Intro
When you run a new job using MissingLink's resourse managment, you'll to provide the job with a git repository for the job to run.

The easiest way to setup to get started is to setup MissingLink's source tracking.
All you need to do is:
* Create a new git repository for code tracking
* Create a `.ml_tracking_repo` file in the root folder project with the repository URL.
* Ensure you have SSH access to Github/Bitbucket.


# Github
For the context of this example, lets assume you're working on your project's repository `git@github.com:your_org/mnist.git`.

### Create a new repository:
* Go to: https://github.com/new
* Choose your company's organization as the owner
* We recommend you name the repository the same as the main repository you're working on with a `_tracking` suffix. 
    *  e.g For `mnist` name it `mnist_tracking`

* After the repository is created; in the repository page click **Clone or download**
* Choose **Use SSH**
* Copy the repository SSH URL
    * It should look like this: `git@github.com:your_org/mnist_tracking.git` 

* Go to your root folder of your main repository, i.e `mnist`.
* Create a new file called `.ml_tracking_repo` and paste the tracking repository URL in it. i.e `git@github.com:your_org/mnist_tracking.git` 

### Ensure you have SSH access to Github

* In your terminal run: `ssh -T git@github.com`, you should see: 
    ```
     > Hi username! You've successfully authenticated, but GitHub does not provide shell access.
     ```
* In case you see an error: `Error: Permission denied` - you'll need add your SSH key to Github.

#### Add your SSH key to Github
> NOTE: in case you don't have already have an SSH key, please follow this [Guide on how to generate a new SSH key](https://help.github.com/en/enterprise/2.15/user/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
* Copy the SSH key to your clipboard by running:
```
# Mac
pbcopy < ~/.ssh/id_rsa.pub

# Windows
clip < ~/.ssh/id_rsa.pub

# OR and copy the output
cat ~/.ssh/id_rsa.pub
```
* Go to: [https://github.com/settings/ssh/new]()
* Paste the SSH key you copied under the **Key** field
* Click **Add SSH key**

* In your terminal re-run: `ssh -T git@github.com`, and you should see the `You've successfully authenticated` message.

### You're all done
* Run your experiment using: `ml run xp`

-----

# Bitbucket
For the context of this example, lets assume you're working on your project's repository `git@bitbucket.org:your_org/mnist.git`.

### Create a new repository:
* Go to: https://bitbucket.org/repo/create
* We recommend you name the repository the same as the main repository you're working on with a `_tracking` suffix. 
    *  e.g For `mnist` name it `mnist_tracking`

* After the repository is created; in the repository page you should see the URL of your repository.
* Change the URL type to **SSH** (instead of **HTTPS**)
* Copy the repository SSH URL
    * It should look like this: `git@bitbucket.org:your_org/mnist_tracking.git` 

* Go to your root folder of your main repository, i.e `mnist`.
* Create a new file called `.ml_tracking_repo` and paste the tracking repository URL in it. i.e `git@bitbucket.org:your_org/mnist_tracking.git` 

### Ensure you have SSH access to Bitbucket

* In your terminal run: `ssh -T git@bitbucket.org`, you should see: 
    ```
     > logged in as username
     ```
* In case you see an error: `Error: Permission denied` - you'll need add your SSH key to Github.

#### Add your SSH key to Bitbucket
> NOTE: in case you don't have already have an SSH key, please follow this [Guide on how to generate a new SSH key](https://confluence.atlassian.com/x/X4FmKw).

* Copy the SSH key to your clipboard by running:
```
# Mac
pbcopy < ~/.ssh/id_rsa.pub

# Windows
clip < ~/.ssh/id_rsa.pub

# OR and copy the output
cat ~/.ssh/id_rsa.pub
```
* Go to your Bitbucket account setting: [https://bitbucket.org/account]()
* Got to **SSH keys** (under **SECURITY** section)
* Click **Add key**
* Paste the SSH key you copied under the **Key** field
* Click **Add key**

* In your terminal re-run: `ssh -T git@bitbucket.org`, and you should see the `logged in as username` message.

### You're all done
* Run your experiment using: `ml run xp`