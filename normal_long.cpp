//program normal

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

const int MAXM = 1000;

double sum[MAXM], ub[MAXM], lb[MAXM];

void getMean() {
	ifstream fin("data_long.txt");
	int n, m;
	fin >> n >> m;
	memset(sum, 0, sizeof(sum));
	for(int i = 0; i < n; i++) {
		double x;
		for(int j = 0; j < m; j++) {
			fin >> x;
			sum[j] += x;
			lb[j] = i ? min(lb[j], x) : x;
			ub[j] = i ? max(ub[j], x) : x;
		}
		fin >> x;
	}
	fin.close();
	for(int i = 0; i < m; i++) {
		sum[i] /= n;
		ub[i] = max(ub[i] - lb[i], 1e-5);
	}
}

void normalize() {
	ifstream fin("data_long.txt");
	ofstream fout("data_long.csv");
	int n, m;
	fin >> n >> m;
	fout << n << ',' << m << endl;
	for(int i = 0; i < n; i++) {
		double x;
		for(int j = 0; j < m; j++) {
			fin >> x;
			fout << (x - sum[j]) / ub[j] << ',';
		}
		fin >> x;
		fout << (int)x << endl;
	}
	cout << endl;
	fin.close();
	fout.close();
}

int main() {
	getMean();
	normalize();
	return 0;
}
