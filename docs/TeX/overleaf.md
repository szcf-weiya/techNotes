# Overleaf

## Integration with Git/GitHub

### Passwordless

Since it does not support SSH key, it requires to enter username and password everytime. Config `credential` as follows,

```bash
git config --global credential.https://git.overleaf.com.username YourOverleafAccount
git config --global credential.https://git.overleaf.com.helper store
```

The second line aims to store the password after the first input.

Finally, no need to enter username and password.

Refer to

- [Why do I have problems with loging in to Overleaf via git](https://stackoverflow.com/questions/59293055/why-do-i-have-problems-with-loging-in-to-overleaf-via-git)
- [gitcredentials - Providing usernames and passwords to Git](https://git-scm.com/docs/gitcredentials)

### Sync with GitHub


Directly operate on Overleaf via clicking the Menu button.

### GitHub as Backup 

![image](https://user-images.githubusercontent.com/13688320/177310131-8cee1767-b4a9-426e-a086-91dd59abad50.png)

If there are frequent updates from local laptop, such as bib files and figures, it might be better to add GitHub as a backup repo.

Suppose you have a project on overleaf.

1. clone it to local laptop.
2. create an empty repo on github, say `NewRepo`
3. add github as another remote repo: `git remote add backup git@github.com:szcf-weiya/NewRepo.git`
4. update from laptop to overleaf, such as uploading figures, updating bib: `git push origin master`
5. pull updates from overleaf: `git pull`
6. backup to github: `git push backup master`

