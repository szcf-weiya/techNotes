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