import click # DO NOT NAME THE FILE AS THE SAME NAME

@click.command()
@click.option("--a", default = 1)
@click.option("--b", is_flag = True, default = False)
@click.option("--c", default = 2, type = float)
def main(a, b, c):
    print(f"a = {a}, b = {b}, c = {c}")

if __name__ == '__main__':
    main()