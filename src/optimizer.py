"""
Parameter Optimization for Exit Signal Screener
다양한 최적화 방법을 제공합니다:
1. Bayesian Optimization (베이지안 최적화)
2. Random Search (랜덤 서치)
3. Genetic Algorithm (유전 알고리즘)
4. Walk-Forward Optimization (워크포워드 최적화)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Callable
from .screener import ExitSignalScreener
import warnings
warnings.filterwarnings('ignore')


class ParameterOptimizer:
    """
    파라미터 최적화 클래스

    튜닝 가능한 파라미터:
    - ma_period: 이동평균선 기간 (5~200)
    - rvol_period: 상대 거래량 기간 (5~50)
    - wick_threshold: 윅 비율 임계값 (0.3~0.8)
    - rvol_threshold: RVOL 임계값 (1.0~5.0)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 ticker: str = "BACKTEST",
                 evaluation_metric: str = 'sharpe_ratio'):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            백테스팅용 OHLCV 데이터
        ticker : str
            종목명
        evaluation_metric : str
            평가 지표: 'sharpe_ratio', 'total_return', 'win_rate', 'profit_factor'
        """
        self.df = df
        self.ticker = ticker
        self.evaluation_metric = evaluation_metric
        self.optimization_history = []

    def evaluate_parameters(self, params: Dict) -> float:
        """
        파라미터 조합의 성능 평가

        Parameters:
        -----------
        params : dict
            {'ma_period': 20, 'rvol_period': 20, 'wick_threshold': 0.5, 'rvol_threshold': 2.0}

        Returns:
        --------
        float : 평가 점수 (높을수록 좋음)
        """
        try:
            # 스크리너 초기화
            screener = ExitSignalScreener(
                ma_period=int(params['ma_period']),
                rvol_period=int(params['rvol_period'])
            )

            # 지표 계산
            filtered_data = screener.apply_filters(self.df.copy())

            # 임계값 커스터마이징
            filtered_data['Custom_Condition_2'] = filtered_data['Wick_Ratio'] >= params['wick_threshold']
            filtered_data['Custom_Condition_3'] = filtered_data['RVOL'] >= params['rvol_threshold']

            filtered_data['Custom_All_Conditions_Met'] = (
                filtered_data['Condition_1_Trend_Breakdown'] &
                filtered_data['Custom_Condition_2'] &
                filtered_data['Custom_Condition_3']
            )

            filtered_data['Custom_Signal'] = np.where(
                filtered_data['Custom_All_Conditions_Met'],
                'SELL',
                'HOLD'
            )

            # 백테스팅 수익률 계산
            returns = self._calculate_strategy_returns(filtered_data)

            # 평가 지표 계산
            score = self._calculate_metric(returns, filtered_data)

            # 히스토리 저장
            self.optimization_history.append({
                **params,
                'score': score,
                'metric': self.evaluation_metric
            })

            return score

        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return -np.inf

    def _calculate_strategy_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        전략 수익률 계산
        SELL 신호 발생 시 다음날 매도한다고 가정
        """
        df['Price_Change'] = df['Close'].pct_change()
        df['Strategy_Position'] = 0  # 0: 보유, 1: 매도(공매도 또는 현금)

        # SELL 신호 발생 시 다음날부터 포지션 종료
        df['Strategy_Position'] = (df['Custom_Signal'] == 'SELL').shift(1).fillna(False).astype(int)

        # 전략 수익률 = 포지션이 0일 때만 수익 발생 (보유 중)
        df['Strategy_Returns'] = df['Price_Change'] * (1 - df['Strategy_Position'])

        return df['Strategy_Returns'].fillna(0)

    def _calculate_metric(self, returns: pd.Series, df: pd.DataFrame) -> float:
        """평가 지표 계산"""
        if self.evaluation_metric == 'sharpe_ratio':
            return self._sharpe_ratio(returns)
        elif self.evaluation_metric == 'total_return':
            return (1 + returns).prod() - 1
        elif self.evaluation_metric == 'win_rate':
            return self._win_rate(df)
        elif self.evaluation_metric == 'profit_factor':
            return self._profit_factor(returns)
        else:
            return self._sharpe_ratio(returns)

    def _sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """샤프 비율"""
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _win_rate(self, df: pd.DataFrame) -> float:
        """승률 (SELL 신호 후 실제로 하락한 비율)"""
        sell_signals = df[df['Custom_Signal'] == 'SELL']
        if len(sell_signals) == 0:
            return 0

        # SELL 신호 다음날 수익률
        next_day_returns = []
        for idx in sell_signals.index:
            try:
                next_idx = df.index.get_loc(idx) + 1
                if next_idx < len(df):
                    next_day_returns.append(df.iloc[next_idx]['Close'] / df.loc[idx, 'Close'] - 1)
            except:
                continue

        if len(next_day_returns) == 0:
            return 0

        # 하락한 비율 (SELL 신호가 맞았던 비율)
        wins = sum(1 for r in next_day_returns if r < 0)
        return wins / len(next_day_returns)

    def _profit_factor(self, returns: pd.Series) -> float:
        """Profit Factor (총 이익 / 총 손실)"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses == 0:
            return gains if gains > 0 else 0

        return gains / losses

    # ============================================================================
    # 1. Random Search (랜덤 서치)
    # ============================================================================

    def random_search(self,
                     n_iterations: int = 100,
                     param_ranges: Dict = None) -> Tuple[Dict, float]:
        """
        랜덤 서치를 통한 파라미터 최적화

        장점:
        - 구현이 간단하고 빠름
        - 그리드 서치보다 효율적 (고차원 공간에서)
        - 병렬화 쉬움

        Parameters:
        -----------
        n_iterations : int
            시도 횟수
        param_ranges : dict
            파라미터 범위 지정

        Returns:
        --------
        best_params : dict
            최적 파라미터
        best_score : float
            최고 점수
        """
        if param_ranges is None:
            param_ranges = {
                'ma_period': (10, 100),
                'rvol_period': (10, 50),
                'wick_threshold': (0.3, 0.8),
                'rvol_threshold': (1.5, 4.0)
            }

        print(f"\n{'='*80}")
        print(f"Random Search Optimization ({n_iterations} iterations)")
        print(f"{'='*80}")

        best_params = None
        best_score = -np.inf

        for i in range(n_iterations):
            # 랜덤 파라미터 생성
            params = {
                'ma_period': np.random.randint(param_ranges['ma_period'][0],
                                              param_ranges['ma_period'][1]),
                'rvol_period': np.random.randint(param_ranges['rvol_period'][0],
                                                param_ranges['rvol_period'][1]),
                'wick_threshold': np.random.uniform(param_ranges['wick_threshold'][0],
                                                   param_ranges['wick_threshold'][1]),
                'rvol_threshold': np.random.uniform(param_ranges['rvol_threshold'][0],
                                                   param_ranges['rvol_threshold'][1])
            }

            score = self.evaluate_parameters(params)

            if score > best_score:
                best_score = score
                best_params = params
                print(f"[Iteration {i+1}/{n_iterations}] New Best! Score: {score:.4f}")
                print(f"  Params: MA={params['ma_period']}, RVOL_Period={params['rvol_period']}, "
                      f"Wick={params['wick_threshold']:.2f}, RVOL_Thresh={params['rvol_threshold']:.2f}")

        print(f"\n{'='*80}")
        print(f"Random Search Complete!")
        print(f"Best Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        print(f"{'='*80}\n")

        return best_params, best_score

    # ============================================================================
    # 2. Bayesian Optimization (베이지안 최적화)
    # ============================================================================

    def bayesian_optimization(self,
                            n_iterations: int = 50,
                            param_ranges: Dict = None,
                            n_initial_points: int = 10) -> Tuple[Dict, float]:
        """
        베이지안 최적화를 통한 파라미터 튜닝

        장점:
        - 적은 시도로 효율적 탐색
        - 이전 결과를 학습하여 다음 탐색 지점 결정
        - 연속형 파라미터에 강함

        주의:
        - scipy 라이브러리 필요 (간단한 구현 버전)

        Parameters:
        -----------
        n_iterations : int
            최적화 반복 횟수
        param_ranges : dict
            파라미터 범위
        n_initial_points : int
            초기 랜덤 탐색 횟수

        Returns:
        --------
        best_params : dict
            최적 파라미터
        best_score : float
            최고 점수
        """
        if param_ranges is None:
            param_ranges = {
                'ma_period': (10, 100),
                'rvol_period': (10, 50),
                'wick_threshold': (0.3, 0.8),
                'rvol_threshold': (1.5, 4.0)
            }

        print(f"\n{'='*80}")
        print(f"Bayesian Optimization ({n_iterations} iterations)")
        print(f"{'='*80}")

        # 초기 랜덤 샘플링
        explored_params = []
        explored_scores = []

        for i in range(n_initial_points):
            params = {
                'ma_period': np.random.randint(param_ranges['ma_period'][0],
                                              param_ranges['ma_period'][1]),
                'rvol_period': np.random.randint(param_ranges['rvol_period'][0],
                                                param_ranges['rvol_period'][1]),
                'wick_threshold': np.random.uniform(param_ranges['wick_threshold'][0],
                                                   param_ranges['wick_threshold'][1]),
                'rvol_threshold': np.random.uniform(param_ranges['rvol_threshold'][0],
                                                   param_ranges['rvol_threshold'][1])
            }

            score = self.evaluate_parameters(params)
            explored_params.append(params)
            explored_scores.append(score)

            print(f"[Initial {i+1}/{n_initial_points}] Score: {score:.4f}")

        best_idx = np.argmax(explored_scores)
        best_params = explored_params[best_idx]
        best_score = explored_scores[best_idx]

        # Expected Improvement 기반 탐색
        for i in range(n_iterations - n_initial_points):
            # 간단한 UCB (Upper Confidence Bound) 전략
            params = self._ucb_next_point(explored_params, explored_scores, param_ranges)

            score = self.evaluate_parameters(params)
            explored_params.append(params)
            explored_scores.append(score)

            if score > best_score:
                best_score = score
                best_params = params
                print(f"[Iteration {i+n_initial_points+1}/{n_iterations}] New Best! Score: {score:.4f}")
                print(f"  Params: MA={params['ma_period']}, RVOL_Period={params['rvol_period']}, "
                      f"Wick={params['wick_threshold']:.2f}, RVOL_Thresh={params['rvol_threshold']:.2f}")

        print(f"\n{'='*80}")
        print(f"Bayesian Optimization Complete!")
        print(f"Best Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        print(f"{'='*80}\n")

        return best_params, best_score

    def _ucb_next_point(self, explored_params: List[Dict],
                       explored_scores: List[float],
                       param_ranges: Dict,
                       exploration_weight: float = 2.0) -> Dict:
        """UCB 기반 다음 탐색 지점 선택"""
        # 간단한 구현: 최고 점수 근처에서 탐색 + 랜덤 탐색
        if len(explored_scores) == 0:
            # 랜덤 선택
            return {
                'ma_period': np.random.randint(param_ranges['ma_period'][0],
                                              param_ranges['ma_period'][1]),
                'rvol_period': np.random.randint(param_ranges['rvol_period'][0],
                                                param_ranges['rvol_period'][1]),
                'wick_threshold': np.random.uniform(param_ranges['wick_threshold'][0],
                                                   param_ranges['wick_threshold'][1]),
                'rvol_threshold': np.random.uniform(param_ranges['rvol_threshold'][0],
                                                   param_ranges['rvol_threshold'][1])
            }

        best_idx = np.argmax(explored_scores)
        best_params = explored_params[best_idx]

        # 최적점 근처에서 가우시안 노이즈로 탐색
        new_params = {
            'ma_period': int(np.clip(
                best_params['ma_period'] + np.random.randn() * 10,
                param_ranges['ma_period'][0],
                param_ranges['ma_period'][1]
            )),
            'rvol_period': int(np.clip(
                best_params['rvol_period'] + np.random.randn() * 5,
                param_ranges['rvol_period'][0],
                param_ranges['rvol_period'][1]
            )),
            'wick_threshold': np.clip(
                best_params['wick_threshold'] + np.random.randn() * 0.05,
                param_ranges['wick_threshold'][0],
                param_ranges['wick_threshold'][1]
            ),
            'rvol_threshold': np.clip(
                best_params['rvol_threshold'] + np.random.randn() * 0.2,
                param_ranges['rvol_threshold'][0],
                param_ranges['rvol_threshold'][1]
            )
        }

        return new_params

    # ============================================================================
    # 3. Genetic Algorithm (유전 알고리즘)
    # ============================================================================

    def genetic_algorithm(self,
                         population_size: int = 20,
                         n_generations: int = 30,
                         mutation_rate: float = 0.2,
                         param_ranges: Dict = None) -> Tuple[Dict, float]:
        """
        유전 알고리즘을 통한 파라미터 최적화

        장점:
        - 전역 최적화에 강함
        - 복잡한 탐색 공간에서 효과적
        - 자연스러운 탐색과 활용의 균형

        Parameters:
        -----------
        population_size : int
            세대당 개체 수
        n_generations : int
            진화 세대 수
        mutation_rate : float
            돌연변이 확률
        param_ranges : dict
            파라미터 범위

        Returns:
        --------
        best_params : dict
            최적 파라미터
        best_score : float
            최고 점수
        """
        if param_ranges is None:
            param_ranges = {
                'ma_period': (10, 100),
                'rvol_period': (10, 50),
                'wick_threshold': (0.3, 0.8),
                'rvol_threshold': (1.5, 4.0)
            }

        print(f"\n{'='*80}")
        print(f"Genetic Algorithm (Population: {population_size}, Generations: {n_generations})")
        print(f"{'='*80}")

        # 초기 개체군 생성
        population = self._initialize_population(population_size, param_ranges)

        best_overall_params = None
        best_overall_score = -np.inf

        for gen in range(n_generations):
            # 각 개체 평가
            fitness_scores = [self.evaluate_parameters(ind) for ind in population]

            # 최고 개체 추적
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_score = fitness_scores[gen_best_idx]
            gen_best_params = population[gen_best_idx]

            if gen_best_score > best_overall_score:
                best_overall_score = gen_best_score
                best_overall_params = gen_best_params
                print(f"[Generation {gen+1}/{n_generations}] New Best! Score: {gen_best_score:.4f}")
                print(f"  Params: MA={gen_best_params['ma_period']}, RVOL_Period={gen_best_params['rvol_period']}, "
                      f"Wick={gen_best_params['wick_threshold']:.2f}, RVOL_Thresh={gen_best_params['rvol_threshold']:.2f}")

            # 선택 (토너먼트 방식)
            selected = self._tournament_selection(population, fitness_scores, population_size)

            # 교차 (Crossover)
            offspring = self._crossover(selected, param_ranges)

            # 돌연변이 (Mutation)
            population = self._mutate(offspring, mutation_rate, param_ranges)

        print(f"\n{'='*80}")
        print(f"Genetic Algorithm Complete!")
        print(f"Best Score: {best_overall_score:.4f}")
        print(f"Best Parameters: {best_overall_params}")
        print(f"{'='*80}\n")

        return best_overall_params, best_overall_score

    def _initialize_population(self, size: int, param_ranges: Dict) -> List[Dict]:
        """초기 개체군 생성"""
        population = []
        for _ in range(size):
            individual = {
                'ma_period': np.random.randint(param_ranges['ma_period'][0],
                                              param_ranges['ma_period'][1]),
                'rvol_period': np.random.randint(param_ranges['rvol_period'][0],
                                                param_ranges['rvol_period'][1]),
                'wick_threshold': np.random.uniform(param_ranges['wick_threshold'][0],
                                                   param_ranges['wick_threshold'][1]),
                'rvol_threshold': np.random.uniform(param_ranges['rvol_threshold'][0],
                                                   param_ranges['rvol_threshold'][1])
            }
            population.append(individual)
        return population

    def _tournament_selection(self, population: List[Dict],
                             fitness_scores: List[float],
                             n_select: int,
                             tournament_size: int = 3) -> List[Dict]:
        """토너먼트 선택"""
        selected = []
        for _ in range(n_select):
            # 랜덤으로 토너먼트 참가자 선택
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected

    def _crossover(self, parents: List[Dict], param_ranges: Dict) -> List[Dict]:
        """교차 (단일점 교차)"""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i+1]

                # 50% 확률로 교차
                if np.random.rand() < 0.5:
                    child1 = {
                        'ma_period': parent1['ma_period'],
                        'rvol_period': parent2['rvol_period'],
                        'wick_threshold': parent1['wick_threshold'],
                        'rvol_threshold': parent2['rvol_threshold']
                    }
                    child2 = {
                        'ma_period': parent2['ma_period'],
                        'rvol_period': parent1['rvol_period'],
                        'wick_threshold': parent2['wick_threshold'],
                        'rvol_threshold': parent1['rvol_threshold']
                    }
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()

                offspring.extend([child1, child2])
            else:
                offspring.append(parents[i].copy())

        return offspring

    def _mutate(self, population: List[Dict],
               mutation_rate: float,
               param_ranges: Dict) -> List[Dict]:
        """돌연변이"""
        for individual in population:
            if np.random.rand() < mutation_rate:
                # 랜덤하게 한 파라미터 변이
                param_to_mutate = np.random.choice(['ma_period', 'rvol_period',
                                                   'wick_threshold', 'rvol_threshold'])

                if param_to_mutate == 'ma_period':
                    individual['ma_period'] = np.random.randint(
                        param_ranges['ma_period'][0],
                        param_ranges['ma_period'][1]
                    )
                elif param_to_mutate == 'rvol_period':
                    individual['rvol_period'] = np.random.randint(
                        param_ranges['rvol_period'][0],
                        param_ranges['rvol_period'][1]
                    )
                elif param_to_mutate == 'wick_threshold':
                    individual['wick_threshold'] = np.random.uniform(
                        param_ranges['wick_threshold'][0],
                        param_ranges['wick_threshold'][1]
                    )
                elif param_to_mutate == 'rvol_threshold':
                    individual['rvol_threshold'] = np.random.uniform(
                        param_ranges['rvol_threshold'][0],
                        param_ranges['rvol_threshold'][1]
                    )

        return population

    # ============================================================================
    # 4. Walk-Forward Optimization (워크포워드 최적화)
    # ============================================================================

    def walk_forward_optimization(self,
                                 train_window: int = 252,  # 1년
                                 test_window: int = 63,     # 3개월
                                 optimization_method: str = 'random_search',
                                 n_iterations: int = 50) -> List[Dict]:
        """
        워크포워드 최적화

        장점:
        - 과최적화 방지
        - 실제 트레이딩 환경에 가까움
        - Out-of-sample 테스트 포함

        Parameters:
        -----------
        train_window : int
            학습 기간 (일수)
        test_window : int
            테스트 기간 (일수)
        optimization_method : str
            사용할 최적화 방법 ('random_search', 'bayesian', 'genetic')
        n_iterations : int
            각 윈도우에서 최적화 반복 횟수

        Returns:
        --------
        results : List[Dict]
            각 윈도우별 최적 파라미터와 성능
        """
        print(f"\n{'='*80}")
        print(f"Walk-Forward Optimization")
        print(f"Train Window: {train_window} days, Test Window: {test_window} days")
        print(f"{'='*80}")

        results = []
        total_length = len(self.df)
        current_start = 0

        window_num = 0

        while current_start + train_window + test_window <= total_length:
            window_num += 1

            # Train/Test 분할
            train_end = current_start + train_window
            test_end = train_end + test_window

            train_data = self.df.iloc[current_start:train_end]
            test_data = self.df.iloc[train_end:test_end]

            print(f"\n[Window {window_num}] Train: {current_start} to {train_end}, Test: {train_end} to {test_end}")

            # Train 데이터로 최적화
            temp_optimizer = ParameterOptimizer(train_data, self.ticker, self.evaluation_metric)

            if optimization_method == 'random_search':
                best_params, train_score = temp_optimizer.random_search(n_iterations=n_iterations)
            elif optimization_method == 'bayesian':
                best_params, train_score = temp_optimizer.bayesian_optimization(n_iterations=n_iterations)
            elif optimization_method == 'genetic':
                best_params, train_score = temp_optimizer.genetic_algorithm(
                    population_size=20,
                    n_generations=n_iterations//20
                )
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")

            # Test 데이터로 검증
            test_optimizer = ParameterOptimizer(test_data, self.ticker, self.evaluation_metric)
            test_score = test_optimizer.evaluate_parameters(best_params)

            print(f"  Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")

            results.append({
                'window': window_num,
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'best_params': best_params,
                'train_score': train_score,
                'test_score': test_score
            })

            # 다음 윈도우로 이동
            current_start += test_window

        print(f"\n{'='*80}")
        print(f"Walk-Forward Optimization Complete! Total Windows: {window_num}")
        print(f"{'='*80}\n")

        # 결과 요약
        self._print_walk_forward_summary(results)

        return results

    def _print_walk_forward_summary(self, results: List[Dict]):
        """워크포워드 결과 요약"""
        print("\n[Walk-Forward Summary]")
        print(f"{'Window':<10}{'Train Score':<15}{'Test Score':<15}{'Difference':<15}")
        print("-" * 60)

        for r in results:
            diff = r['test_score'] - r['train_score']
            print(f"{r['window']:<10}{r['train_score']:<15.4f}{r['test_score']:<15.4f}{diff:<15.4f}")

        avg_train = np.mean([r['train_score'] for r in results])
        avg_test = np.mean([r['test_score'] for r in results])

        print("-" * 60)
        print(f"{'Average':<10}{avg_train:<15.4f}{avg_test:<15.4f}{avg_test - avg_train:<15.4f}")

    def export_optimization_history(self, filename: str = 'optimization_history.csv'):
        """최적화 히스토리를 CSV로 저장"""
        if len(self.optimization_history) == 0:
            print("No optimization history to export.")
            return

        df = pd.DataFrame(self.optimization_history)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nOptimization history saved to: {filename}")
        print(f"Total evaluations: {len(df)}")


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')

    sample_data = pd.DataFrame({
        'Open': 100 + np.random.randn(500).cumsum() * 0.5,
        'Close': 100 + np.random.randn(500).cumsum() * 0.5,
        'Volume': np.random.randint(1000000, 5000000, 500)
    })

    sample_data['High'] = sample_data[['Open', 'Close']].max(axis=1) + abs(np.random.randn(500) * 2)
    sample_data['Low'] = sample_data[['Open', 'Close']].min(axis=1) - abs(np.random.randn(500) * 2)
    sample_data.index = dates

    # 옵티마이저 초기화
    optimizer = ParameterOptimizer(sample_data, ticker="TEST", evaluation_metric='sharpe_ratio')

    # 1. Random Search
    print("\n\n### Running Random Search ###")
    best_params_random, score_random = optimizer.random_search(n_iterations=30)

    # 2. Bayesian Optimization
    print("\n\n### Running Bayesian Optimization ###")
    best_params_bayes, score_bayes = optimizer.bayesian_optimization(n_iterations=30)

    # 3. Genetic Algorithm
    print("\n\n### Running Genetic Algorithm ###")
    best_params_genetic, score_genetic = optimizer.genetic_algorithm(
        population_size=15,
        n_generations=10
    )

    # 히스토리 저장
    optimizer.export_optimization_history('optimization_history.csv')
