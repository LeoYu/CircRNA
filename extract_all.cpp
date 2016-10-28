//program extract_all

#include<iostream>
#include<fstream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<algorithm>
#include<set>
#include<map>
#include<vector>

using namespace std;

struct Segment {
	string tag;
	int l, r;
	char sign;
	bool label;
};

struct Alu {
	string tag, typ;
	int l, r;
};

bool operator < (Segment a, Segment b) {
	if(a.tag != b.tag)
		return a.tag < b.tag;
	if(a.l != b.l)
		return a.l < b.l;
	if(a.r != b.r)
		return a.r < b.r;
	return a.sign && !b.sign;
}

bool operator < (Alu a, Alu b) {
	if(a.tag != b.tag)
		return a.tag < b.tag;
	if(a.l != b.l)
		return a.l < b.l;
	return a.r < b.r;
}

const int N = 300000;
const int K = 3;
const int KMER = 4 + 16 + 64;
const int ALU = 37;
const int M = 1 + KMER;// + ALU;
const int MAXN = 600000;
const int MAXALU = 1200000;
const int MAXLEN = 100000;

int total = 0;
int total_alu = 0;
Segment data[MAXN];
Alu alu[MAXALU];
string alutyp[MAXALU];
int num[MAXN];
int flag[MAXN];
char q[MAXLEN], seq[MAXLEN];
double freq[KMER];

/* randomly draw N samples from both classes */

int gettypnum(string s) {
	return lower_bound(alutyp, alutyp + ALU, s) - alutyp;
}

void getIndex() {
	ifstream fpos("pos_index.txt");
	ifstream fneg("neg_index.txt");
	string tag;
	int l, r;
	char sign;
	while(fpos >> tag) {
		fpos >> l >> r >> sign;
		data[total++] = (Segment){tag, l, r, sign, true};
	}
	random_shuffle(data, data + total);
	total = min(total, N);
	int temp = total;
	while(fneg >> tag) {
		fneg >> l >> r >> sign;
		data[total++] = (Segment){tag, l, r, sign, false};
	}
	random_shuffle(data + temp, data + total);
	total = min(total, temp + N);
	sort(data, data + total);
	fpos.close();
	fneg.close();
}

void getAlu() {
	ifstream falu("hg19_Alu.bed");
	string tag, typ;
	int l, r;
	while(falu >> tag >> l >> r >> typ) {
		alutyp[total_alu] = typ;
		alu[total_alu++] = (Alu){tag, typ, l, r};
		falu >> tag >> typ;
	}
	falu.close();
	sort(alu, alu + total_alu);
	sort(alutyp, alutyp + total_alu);
	cout << "Total Alu: " << unique(alutyp, alutyp + total_alu) - alutyp << endl;
}

void getData() {
	static char comp[128];
	comp['A'] = 'T';
	comp['C'] = 'G';
	comp['G'] = 'C';
	comp['T'] = 'A';
	static int number[128];
	number['A'] = 0;
	number['C'] = 1;
	number['G'] = 2;
	number['T'] = 3;

	ifstream fstr("hg19.fa");
	ofstream samp("data_all.txt");
	string s, tag;
	int cnt = 0, active, start;
	samp << total << ' ' << M << endl;
	while(fstr >> s) {
		if(s.size() >= 1 && s[0] == '>') {
			tag = s.substr(1);
			cnt = active = 0;
			start = lower_bound(data, data + total, (Segment){tag, -1, -1, ' '}) - data;
			cout<<"tag = "<<tag<<endl;
			continue;
		}
		for(int i = 0; i < s.size(); i++) {
			while(start < total && data[start].tag == tag && data[start].l < cnt)
				start++;
			while(start < total && data[start].tag == tag && data[start].l == cnt)
				{flag[start] = 1;num[active++] = start++;}
			char c = s[i];
			if(c >= 'a' && c <= 'z')
				c -= 32;
			q[cnt % MAXLEN] = c;
			cnt++;
			for(int j = 0; j < active; j++)
				if(data[num[j]].r == cnt) {
					int t = num[j];
					flag[t] = 2;
					active--;
					swap(num[j], num[active]);
					j--;

					int len = data[t].r - data[t].l;
					int head = (cnt - len) % MAXLEN;
					int tail = (cnt - 1) % MAXLEN;
					for(int k = 0; k < len; k++)
						if(data[t].sign == '+') {
							seq[k] = q[head++];
							if(head == MAXLEN)
								head = 0;
						} else {
							seq[k] = comp[q[tail--]];
							if(tail < 0)
								tail = MAXLEN - 1;
						}

					/* output length */
					samp << len << ' ';

					/* output k-mer data */
					memset(freq, 0, sizeof(freq));
					for(int k = 0; k < len; k++) {
						int mask = 0, cur = 0;
						for(int l = 0; l <= min(k, K - 1); l++) {
							mask |= number[seq[k - l]] << (l * 2);
							freq[cur + mask] += 1.0 / (len - l);
							cur += 4 << (l * 2);
						}
					}
					for(int k = 0; k < KMER; k++)
						samp << freq[k] << ' ';

					/* output Alu data */
					/*memset(freq, 0, sizeof(freq));
					Alu temp1 = {tag, "", data[t].l, -1};
					Alu temp2 = {tag, "", data[t].r, -1};
					int lb = lower_bound(alu, alu + total_alu, temp1) - alu;
					int ub = lower_bound(alu, alu + total_alu, temp2) - alu;
					for(int k = lb - 1; k < ub; k++)
						if(k >= 0 && k < total_alu && alu[k].tag == tag)
							if(max(alu[k].l, data[t].l) < min(alu[k].r, data[t].r))
								freq[gettypnum(alu[k].typ)]++;
					for(int k = 0; k < ALU; k++)
						samp << freq[k] << ' ';*/

					/* output label */
					samp << data[t].label << endl;
				}
		}
	}
	fstr.close();
	samp.close();
	for(int i = 0; i < total; i++)
		if(flag[i] < 2)
			cout<<flag[i]<<' '<<data[i].tag<<' '<<data[i].l <<' '<<data[i].r<<' '<<data[i].sign<<' '<<data[i].label;
}

int main() {
	getIndex();
	getAlu();
	getData();
	return 0;
}
