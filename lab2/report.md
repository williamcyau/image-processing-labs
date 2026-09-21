# Lab 2 Report

Name: William Yau 

Github repo: https://github.com/williamcyau/image-processing-labs.git

## D1

The `xxd` output of my `hello.txt`:

```
00000000: 4865 6c6c 6f2c 2077 6f72 6c64 0a         Hello, world.
```

The bytes are hexidecimal ASCII indices indicating the characters typed: `Hello, world.`.

## D2

`hello.txt` has 13 bytes. `hello.docx` has 13311 bytes. They differ so much because `hello.docx` defines much more than just the characters, but also how the characters appear in Microsoft Word, such as fonttype, fontsize, etc.

## D3

The output of `python char_count.py hello.txt`:

```
hello.txt contains 13 bytes
byte 10 = hex 0a = '\n' occurs 1 times
byte 32 = hex 20 = ' ' occurs 1 times
byte 44 = hex 2c = ',' occurs 1 times
byte 72 = hex 48 = 'H' occurs 1 times
byte 100 = hex 64 = 'd' occurs 1 times
byte 101 = hex 65 = 'e' occurs 1 times
byte 108 = hex 6c = 'l' occurs 3 times
byte 111 = hex 6f = 'o' occurs 2 times
byte 114 = hex 72 = 'r' occurs 1 times
byte 119 = hex 77 = 'w' occurs 1 times
```

The output of `ls -l hello.txt`:

```
-rw-r--r--  1 yauckwilliam  staff  13 Sep  4 02:20 hello.txt
```

## D4

The first lines of `xxd kodim23.pgm | head`:

```
00000000: 5035 0a37 3638 2035 3132 0a32 3535 0a72  P5.768 512.255.r
00000010: 7376 7476 7673 7876 7679 7777 7676 7775  svtvvsxvvywwvvwu
00000020: 7373 7373 7072 6d69 6867 6057 534e 4f51  ssssprmihg`WSNOQ
00000030: 4f4c 4e4e 4e4e 4c4c 4b4c 494c 4d4c 4c4c  OLNNNNLLKLILMLLL
00000040: 494c 4a49 4c4c 4948 4b49 4b4c 4b49 4949  ILJILLIHKIKLKIII
00000050: 4949 494b 4949 4848 494b 4847 4949 4949  IIIKIIHHIKHGIIII
00000060: 4847 4949 4949 4849 494b 494b 4948 494b  HGIIIIHIIKIKIHIK
00000070: 4b4c 4a4a 4b4f 4f50 4f4f 5253 5151 5352  KLJJKOOPOORSQQSR
00000080: 5350 5150 5051 504f 4d4c 4e4c 4d4d 4d4c  SPQPPQPOMLNLMMML
00000090: 4a4c 4a49 4a49 4846 4649 4643 4545 4643  JLJIJIHFFIFCEEFC
```

The header is: `5035 0a37 3638 2035 3132 0a32 3535 0a` meaning `P5.768 512.255.`.

## D5

The header occupies `30` hexadecimal digits, i.e., `15` bytes.

The total number of pixels is `768*512 = 393216`, i.e., `393216` bytes.

Therefore, the file size should be `393216 + 15 = 393231` bytes.

What `ls -l kodim23.pgm` returns:

```
-rw-r--r--  1 yauckwilliam  staff  393231 Sep  4 02:43 kodim23.pgm
```

This indicates `kodim23.pgm` indeed has `393231` bytes.

## Claude use

I used to Claude to help me convert this markdown file to pdf.
