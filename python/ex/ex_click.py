import click # DO NOT NAME THE FILE AS THE SAME NAME

@click.command()
@click.option("--a", default = 1)
@click.option("--b", is_flag = True, default = False)
def main(a, b):
    print(f"a = {a}, b = {b}")

if __name__ == '__main__':
    main()