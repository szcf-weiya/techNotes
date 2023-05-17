// refer: https://www.zhihu.com/question/393121310/answer/1606381900
// NOTE: You may need to change the magic code.
#include <stdio.h>

int main(int argc, char * argv[])
{
	if (argc != 3)
	{
		printf("Usage: %s <input filename> <output filename>\n", argv[0]);
		return 0;
	}

	FILE* fin;

	fin = fopen(argv[1], "rb");

	if (fin == NULL)
	{
		printf("Can not open %s\n", argv[1]);
		return -1;
	}

	FILE* fout;

	fout = fopen(argv[2], "wb");

	if (fout == NULL)
	{
		printf("Can not open %s\n", argv[2]);
		fclose(fin);
		return -1;
	}

	while (!feof(fin))
	{
		unsigned char b;

		b = fgetc(fin);

        fputc(b ^ 0xeb, fout);
	}

	fclose(fin);
	fclose(fout);

	return 0;
}
